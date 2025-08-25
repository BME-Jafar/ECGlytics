import numpy as np
import base64
import xml.etree.ElementTree as ET
import pandas as pd


def base64decode(data: str) -> np.ndarray:
    """
    Decodes a Base64-encoded ECG signal, reads from a file, and converts it to a NumPy array.

    Args:
        file_path (str): Base64-encoded ECG waveform data.

    Returns:
        np.ndarray: Decoded waveform as a NumPy array.
    """
    # Decode Base64 data
    decoded_data = base64.b64decode(data)

    # Convert to int8 and apply modulo operation
    # Convert to int8 (signed)
    decoded_data = np.frombuffer(decoded_data, dtype=np.int8)
    # Apply modulo to ensure uint8 range
    decoded_data = np.mod(decoded_data, 256)
    decoded_data = decoded_data.astype(np.uint8)  # Ensure it's uint8

    return decoded_data


def xml_to_dict(element: ET.Element) -> dict:
    """
    Recursively converts an XML element and its children into a dictionary.

    Args:
        element (ET.Element): Root XML element.

    Returns:
        dict: Dictionary representation of the XML tree.
    """
    data = {element.tag: {} if element.attrib else None}

    # Add attributes
    if element.attrib:
        data[element.tag].update(("@" + k, v)
                                 for k, v in element.attrib.items())

    # Recursively add child elements
    children = list(element)
    if children:
        child_data = {}
        for child in children:
            child_dict = xml_to_dict(child)
            for key, value in child_dict.items():
                if key in child_data:
                    if not isinstance(child_data[key], list):
                        child_data[key] = [child_data[key]]
                    child_data[key].append(value)
                else:
                    child_data[key] = value
        data[element.tag] = child_data
    else:
        data[element.tag] = element.text.strip() if element.text else None

    return data


def parse_xml_to_dict(file_path: str) -> dict:
    """
    Parses an XML file and converts it into a dictionary.

    Args:
        file_path (str): Path to the XML file.

    Returns:
        dict: Dictionary representation of the XML data.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    return xml_to_dict(root)


def GEXMLparser(path_To_XML: str, number_of_leads: int = 12):
    """
    Extracts ECG waveform data and metadata from parsed GE MUSE XML data.

    Args:
        path_To_XML (str):  Path to a single XML file, must contain extension.
        number_of_leads (int): Number of ECG leads (12 or 8). Defaults to 12.

    Returns:
        df_median (pd.DataFrame): DataFrame containing median waveforms.
        df_rhythm (pd.DataFrame): DataFrame containing rhythm waveforms.
        metadata (dict): Metadata information.
        measurement_data (dict): Measurement annotations if available.
    """
    try:
        data = parse_xml_to_dict(path_To_XML)
    except:
        print(f"Error Reading from {path_To_XML}!")
        return None
    patient_id = data[[*data][0]]['PatientDemographics']['PatientID']
    metadata = data[[*data][0]]['PatientDemographics']
    metadata["samplingFrequency"] = int(
        data[[*data][0]]['Waveform'][0]["SampleBase"])
    metadata["AcquisitionDate"] = data[[*data][0]
                                       ]['TestDemographics']['AcquisitionDate']
    metadata["AcquisitionTime"] = data[[*data][0]
                                       ]['TestDemographics']['AcquisitionTime']

    lead_order_12 = ["I", "II", "III", "V1", "V2",
                     "V3", "V4", "V5", "V6", "aVR", "aVL", "aVF"]
    lead_order_8 = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]

    beat_median = {}
    beat_rhythm = {}

    for waveform in data[[*data][0]]["Waveform"]:
        file_suffix = "Median" if waveform['WaveformType'] == 'Median' else "Rhythm"
        for lead_data in waveform["LeadData"]:
            lead_id = lead_data["LeadID"]
            amplitude_units = float(
                lead_data["LeadAmplitudeUnitsPerBit"].replace(",", "."))
            wave = np.round(np.frombuffer(base64decode(
                lead_data["WaveFormData"]), dtype=np.int16) * amplitude_units)

            if file_suffix == "Median":
                beat_median[lead_id] = wave
            else:
                beat_rhythm[lead_id] = wave

    if number_of_leads == 12:
        for beat in [beat_median, beat_rhythm]:
            if beat:
                beat["III"] = np.round(beat.get("II", 0) - beat.get("I", 0))
                beat["aVR"] = np.round(-0.5 *
                                       (beat.get("II", 0) + beat.get("I", 0)))
                beat["aVL"] = np.round(
                    beat.get("I", 0) - 0.5 * beat.get("II", 0))
                beat["aVF"] = np.round(
                    beat.get("II", 0) - 0.5 * beat.get("I", 0))

    df_median = pd.DataFrame(
        beat_median, columns=lead_order_12 if number_of_leads == 12 else lead_order_8)
    df_rhythm = pd.DataFrame(
        beat_rhythm, columns=lead_order_12 if number_of_leads == 12 else lead_order_8)

    return df_median, df_rhythm, metadata


def mortaraXMLPARSER(path_To_XML: str):
    # Load XML
    tree = ET.parse(path_To_XML)
    root = tree.getroot()

    # Locate waveform section
    cycle = root.find("TYPICAL_CYCLE")
    units_per_mv = int(cycle.attrib.get("UNITS_PER_MV", "1000"))
    bits = int(cycle.attrib.get("BITS", "16"))
    sample_freq = int(cycle.attrib.get("SAMPLE_FREQ", "500"))

    signals = {}
    for ch in cycle.findall("TYPICAL_CYCLE_CHANNEL"):
        name = ch.attrib["NAME"]
        raw_b64 = ch.items()[-1][-1]
        raw_bytes = base64.b64decode(raw_b64)
        # interpret as signed 16-bit
        data = np.frombuffer(raw_bytes, dtype=np.int16)

        # convert to mV
        data_mv = data / units_per_mv
        signals[name] = data_mv

    df_median = pd.DataFrame(signals)
    if cycle is not None:
        root.remove(cycle)
    for ch in root.findall("CHANNEL"):
        name = ch.attrib["NAME"]
        raw_b64 = ch.items()[-1][-1]
        raw_bytes = base64.b64decode(raw_b64)
        # interpret as signed 16-bit
        data = np.frombuffer(raw_bytes, dtype=np.int16)
        # convert to mV
        data_mv = data / units_per_mv
        signals[name] = data_mv

    df_rythm = pd.DataFrame(signals)

    return sample_freq, df_median, df_rythm

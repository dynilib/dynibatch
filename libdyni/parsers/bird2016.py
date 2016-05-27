import os
import xmltodict


def get_label(audio_relpath, xml_root):
    xml_path = os.path.join(
        xml_root,
        os.path.splitext(os.path.basename(audio_relpath))[0],
        ".xml")
    with open(xml_path, "rb") as f:
        return xmltodict.parse(f)["Audio"]["ClassId"]

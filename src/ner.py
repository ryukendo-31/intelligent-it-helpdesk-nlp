import re


PATTERNS = {
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",

    "IP_ADDRESS": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",

    "WINDOWS_VERSION": r"\bWindows\s+(?:10|11|7|8(?:\.1)?)\b",

    "MACOS_VERSION": r"\bmacOS\s+[A-Za-z0-9 .-]+\b",

    "SOFTWARE_VERSION": r"\b(?:Zoom|Chrome|Firefox|Microsoft\s+Office|Outlook|Teams)\s+\d+(?:\.\d+)+\b",

    "SOFTWARE": r"\b(?:Zoom|Chrome|Firefox|Microsoft\s+Office|Outlook|Teams|Windows|macOS|Android|iOS)\b",

    "DEVICE_MODEL": r"\b(?:Dell|HP|Lenovo|Apple|Samsung|Microsoft|Asus|Acer)\s+[A-Za-z0-9]+(?:\s+[A-Za-z0-9]+){0,3}\b",

    "DEVICE_TYPE": r"\b(?:laptop|desktop|computer|keyboard|mouse|monitor|screen|headphones|headset|speaker|speakers|printer|scanner|phone|tablet|router|modem|charger|webcam|microphone)\b",

    "NETWORK_ENTITY": r"\b(?:Wi[- ]?Fi|Bluetooth|Ethernet|VPN|LAN|WAN)\b"
}


def extract_entities(text):

    text = str(text)

    entities = []

    for label, pattern in PATTERNS.items():

        matches = re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for match in matches:

            entities.append({
                "text": match.group(),
                "label": label,
                "start": match.start(),
                "end": match.end()
            })

    entities.sort(
        key=lambda x: x["start"]
    )

    unique_entities = []

    seen = set()

    for entity in entities:

        key = (
            entity["text"].lower(),
            entity["label"]
        )

        if key not in seen:

            seen.add(key)

            unique_entities.append(
                entity
            )

    return unique_entities


if __name__ == "__main__":

    text = input(
        "Enter ticket text: "
    )

    entities = extract_entities(
        text
    )

    print("\n" + "=" * 60)
    print("NAMED ENTITY EXTRACTION")
    print("=" * 60)

    if not entities:

        print(
            "No entities found."
        )

    else:

        for entity in entities:

            print(
                f"{entity['text']} "
                f"-> {entity['label']}"
            )
import re


def is_version_below(target_version: str, comparison_version: str) -> bool:
    """
    Compare two version strings and return True if the target version is below the comparison version.

    The comparison is done by comparing the numeric parts of the version strings.
    If the numeric parts are equal, the comparison is done by comparing the suffixes.

    :param target_version: The version to compare
    :param comparison_version: The version to compare against
    :return: True if the target version is below the comparison version
    """

    # Normalize and extract numeric parts of version strings
    def normalize_and_extract(version: str) -> tuple[list[int], str]:
        """
        Normalize the version string and extract the numeric parts and the suffix.
        """
        # Remove leading 'v' or 'v.'
        clean_version = re.sub(r"^v\.?|^v?", "", version)
        # Split version into numeric parts and suffix
        parts = re.split(r"[-+]", clean_version, 1)
        numeric_parts = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""
        # Convert numeric parts into list of integers
        numeric_list = [int(n) for n in numeric_parts.split(".") if n.isdigit()]
        return numeric_list, suffix

    target_parts, target_suffix = normalize_and_extract(target_version)
    comparison_parts, comparison_suffix = normalize_and_extract(comparison_version)

    # Compare the numeric parts
    if target_parts == comparison_parts:
        return bool(target_suffix and not comparison_suffix)

    return target_parts < comparison_parts

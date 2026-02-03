from typing import List, Tuple


def split_message_stream(
    current_buffer: str, new_chunk: str, min_length: int = 400
) -> Tuple[List[str], str]:
    """
    Split message stream into parts based on \\n\\n separator with a minimum length threshold.

    Args:
        current_buffer: The current accumulated message buffer
        new_chunk: New chunk of text to append
        min_length: Minimum length for a part to be split out (default: 400)

    Returns:
        tuple: (list of ready parts to send, remaining buffer)
    """
    buffer = current_buffer + new_chunk
    parts = buffer.split("\n\n")

    # If there's only one part (no separator found), keep accumulating
    if len(parts) == 1:
        return [], buffer

    ready_parts: List[str] = []
    remaining_buffer = ""

    # Process all parts except the last one (which might be incomplete)
    for part in parts[:-1]:
        if not part:  # Skip empty parts
            continue

        # If we have a remaining buffer, check if combining makes sense
        if remaining_buffer:
            combined = remaining_buffer + "\n\n" + part
            if len(combined) >= min_length:
                ready_parts.append(combined)
                remaining_buffer = ""
            else:
                # Keep accumulating if under threshold
                remaining_buffer = combined
        else:
            # Start new buffer or emit if long enough
            if len(part) >= min_length:
                ready_parts.append(part)
            else:
                remaining_buffer = part

    # Handle the last part (potentially incomplete)
    last_part = parts[-1]
    if remaining_buffer:
        remaining_buffer = (
            remaining_buffer + "\n\n" + last_part if last_part else remaining_buffer
        )
    else:
        remaining_buffer = last_part

    return ready_parts, remaining_buffer


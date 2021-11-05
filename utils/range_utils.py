from typing import Tuple, Callable, Any, List
import portion as P


def merge_range(input_offset: Tuple[int, int], target_offset: Tuple[int, int]):
    """
    Merge two ranges
    :param input_offset: input offset tuple
    :param target_offset: output offset tuple
    :return: new merged range
    """
    input_range = P.closedopen(input_offset[0], input_offset[1])
    target_range = P.closedopen(target_offset[0], target_offset[1])
    return input_range | target_range


def range_diff(input_offset: Tuple[int, int], target_offset: Tuple[int, int]):
    """
    Calculate the range difference between input offset and target_offset
    :param input_offset: input offset tuple
    :param target_offset: output offset tuple
    :return: range difference
    """
    input_range = P.closedopen(input_offset[0], input_offset[1])
    target_range = P.closedopen(target_offset[0], target_offset[1])
    return input_range - target_range


def is_overlapped_range(input_offset: Tuple[int, int], target_offset: Tuple[int, int]) -> bool:
    """
    Check if two offset ranges are overlapped
    :param input_offset: input offset tuple
    :param target_offset: output offset tuple
    :return: if two ranges are overlapped
    """
    if input_offset[0] > input_offset[1] or target_offset[0] > target_offset[1]:
        return False
    input_range = P.closedopen(input_offset[0], input_offset[1])
    target_range = P.closedopen(target_offset[0], target_offset[1])
    return input_range.overlaps(target_range)


def is_equal_range(input_offset: Tuple[int, int], target_offset: Tuple[int, int]) -> bool:
    """
    Check if two offset ranges are the same
    :param input_offset: input offset tuple
    :param target_offset: output offset tuple
    :return: if two ranges are the same
    """
    if input_offset[0] > input_offset[1] or target_offset[0] > target_offset[1]:
        return False
    input_range = P.closedopen(input_offset[0], input_offset[1])
    target_range = P.closedopen(target_offset[0], target_offset[1])
    return input_range == target_range


def is_in_range(input_offset: Tuple[int, int], target_offset: Tuple[int, int]) -> bool:
    """
    Check if one offset range contains the other
    :param input_offset: input offset
    :param target_offset: in target offset
    :return: if is overlapped
    """
    if input_offset[0] > input_offset[1] or target_offset[0] > target_offset[1]:
        return False
    input_range = P.closedopen(input_offset[0], input_offset[1])
    target_range = P.closedopen(target_offset[0], target_offset[1])
    return input_range in target_range


def filter_repeated(items: List[Any], start_key: Callable[[Any], int], end_key: Callable[[Any], int]):
    """
    Filter items with self contained offsets from list
    :param items: item list
    :param start_key: callback function to get start
    :param end_key: callback function to get end
    :return: items containing and contained
    """
    def find_unique_range_items():
        """
        Find longer and shorter offset items which one contains or contained by another
        """
        # Next item range contains item range (next item range is longer)
        if is_in_range(item_offset, next_item_offset):
            # Remove item range from longer range items
            if item_offset in longer_offset_items:
                del longer_offset_items[item_offset]
            # Remove next item range from shorter range items
            if next_item_offset in shorter_offset_items:
                del shorter_offset_items[next_item_offset]
            # Add next item range to longer range items
            if next_item_offset not in longer_offset_items:
                longer_offset_items[next_item_offset] = next_item
            # Add item range to shorter range items
            if item_offset not in shorter_offset_items:
                shorter_offset_items[item_offset] = item
        # Item range contains next item range (Item range is longer)
        elif is_in_range(next_item_offset, item_offset):
            # Remove next item range from longer range items
            if next_item_offset in longer_offset_items:
                del longer_offset_items[next_item_offset]
            # Remove item range from shorter range items
            if item_offset in shorter_offset_items:
                del shorter_offset_items[item_offset]
            # Add item range to longer range items
            if item_offset not in longer_offset_items:
                longer_offset_items[item_offset] = item
            # Add next item range to shorter range items
            if next_item_offset not in shorter_offset_items:
                shorter_offset_items[next_item_offset] = next_item
        else:
            # Item range and next item range does not contain each other, add both
            if item_offset not in longer_offset_items and item_offset not in shorter_offset_items:
                longer_offset_items[item_offset] = item
                shorter_offset_items[item_offset] = item
            if next_item_offset not in longer_offset_items and next_item_offset not in shorter_offset_items:
                longer_offset_items[next_item_offset] = next_item
                shorter_offset_items[next_item_offset] = next_item

    # Save longer and shorter range items
    longer_offset_items, shorter_offset_items = {}, {}
    if items:
        if len(items) == 1:
            # Get item offset through callback function
            item_offset = (start_key(items[0]), end_key(items[0]))
            # Append to both range items if length of items is 1
            longer_offset_items[item_offset] = items[0]
            shorter_offset_items[item_offset] = items[0]
        else:
            for i, item in enumerate(items):
                # Get item range through callback function
                item_offset = (start_key(item), end_key(item))
                for next_item in items[i + 1:]:
                    # Get next item offset through callback function
                    next_item_offset = (start_key(next_item), end_key(next_item))
                    # find longer and shorter range items
                    find_unique_range_items()

    return list(longer_offset_items.values()), list(shorter_offset_items.values())

import copy
from typing import List, Dict, Tuple, Any
from pyconst import Const

from . import grpc_client
from utils.range_utils import merge_range, is_in_range, is_overlapped_range

# Recognizers text result type
RECOGNIZER_TYPE = Const(
    ("VALUE", 1, "value"),
    ("TIME", 2, "time")
)
# Value types for Recognizers Text to recognize
RECOGNIZE_VALUE_TYPES = {"number", "ordinal", "percentage", "numberrange", "age",
                         "dimension", "currency", "temperature"}

class Recognizer:
    """
    Recognizers Text Entity (with raw and processed results from Recognizers Text)
    """
    # Raw result from recognizers text
    raw_res: List[Any] = None
    # Processed result by removing repeated time in value
    proc_res: Dict[str, Any] = None

    def __init__(self):
        pass

    @staticmethod
    def get_type_name(type_name: str) -> str or None:
        """
        Math recognizers text type names to value or time
        :param type_name: recognizers text raw type name
        :return: value or time
        """
        # Match value
        if type_name in RECOGNIZE_VALUE_TYPES:
            return RECOGNIZER_TYPE.VALUE.label
        # Match time
        elif type_name.split('.')[0] == "datetimeV2":
            return RECOGNIZER_TYPE.TIME.label
        else:
            return None

    @ staticmethod
    def run_recognizer(text: str):
        """
        Run recognizers text through GPRC Service
        :param text: input text
        :return: recognizers text result
        """
        res = grpc_client.run_recognizer(text)
        return res

    def process_result(self, rec_res: List[Any], text: str = None):
        """
        Filer repeat recognizer items based on offset range, larger ranged offset remains
        :param rec_res: recognized result
        :param text: sentence text
        :return: self
        """
        def get_resolution(rec_item, key):
            """
            Get value based on key from resolution
            :param rec_item: current recognized item
            :param key: key from resolution in recognized item
            """
            if 'Resolution' in rec_item and rec_item['Resolution']:
                if key in rec_item['Resolution'] and rec_item['Resolution'][key]:
                    return rec_item['Resolution'][key]
            return None

        def has_empty_value(rec_item):
            """
            Check if current recognized item has an empty value
            :param rec_item: current recognized item
            :return: True or False
            """
            # Get recognized item type (value or time)
            rec_type = self.get_type_name(rec_item['TypeName'])
            # Check the value from the current recognized item is empty
            if rec_type == RECOGNIZER_TYPE.VALUE.label:
                if 'Resolution' in rec_item and rec_item['Resolution']:
                    if 'value' in rec_item['Resolution'] and rec_item['Resolution']['value'] is None:
                        return True
            return False

        def has_wrong_inch(rec_item):
            """
            Check if current recognized item mis-recognized Unit 'Inch'
            :param rec_item: current recognized item
            :return: True or False
            """
            if rec_item:
                # Unit exists and is 'Inch'
                if get_resolution(rec_item, 'unit') == "Inch":
                    # Text exists and contains 'in'
                    if "Text" in rec_item and "in" in rec_item["Text"]:
                        return True
            return False

        def has_wrong_negative(rec_item):
            """
            Check if current recognized item mis-recognized a negative value
            :param rec_item: current recognized item
            :return: True or False
            """
            if rec_res:
                # First character of recognized item text contains '-'
                if 'Text' in rec_item and rec_item['Text'] is not None and rec_item['Text'][0] == '-':
                    if 'Start' in rec_item and rec_item['Start'] is not None:
                        if rec_item['Start'] > 0:
                            # The previous character is not empty
                            if text[rec_item['Start']-1] != '':
                                return True
            return False

        # Save recognizers text result
        self.raw_res = rec_res
        # Init processed result data structure
        self.proc_res = {
            RECOGNIZER_TYPE.VALUE.label: [],
            RECOGNIZER_TYPE.TIME.label: [],
        }
        # Save global modified recognized items
        mod_rec_items = {}
        if rec_res:
            for i, recognized_item in enumerate(rec_res):
                keep = True
                # Start and end offset from recognizers text is both closed, we use closed open
                offset = (recognized_item['Start'], recognized_item['End'] + 1)
                # Compare with the rest of the recognized items
                for j, next_recognized_item in enumerate(self.raw_res[i+1:]):
                    next_offset = (next_recognized_item['Start'], next_recognized_item['End'] + 1)
                    next_type_name = self.get_type_name(next_recognized_item['TypeName'])
                    # If type of next item is matched and current offset is in next offset, remove the current item
                    if next_type_name:
                        if is_in_range(offset, next_offset):
                            # Skip next recognized items where Unit is recognized incorrectly
                            if not has_wrong_inch(next_recognized_item):
                                keep = False
                                break
                        elif is_overlapped_range(offset, next_offset):
                            # Check if next recognized item is a number range
                            if 'TypeName' in next_recognized_item and next_recognized_item['TypeName'] == 'numberrange':
                                # Get unit of current recognized unit
                                unit = None
                                # Percent and number range cannot co-exist, we treat percentage as unit for ranges
                                if not has_wrong_inch(recognized_item):
                                    if 'TypeName' in recognized_item and recognized_item['TypeName'] == 'percentage':
                                        unit = "%"
                                    else:
                                        unit = get_resolution(recognized_item, 'unit')

                                # Merge into number range, text is currently not merged
                                mod_next_rec_item = copy.deepcopy(next_recognized_item)
                                merged_range = merge_range(offset, next_offset)
                                if merged_range is not None:
                                    if 'Start' in mod_next_rec_item:
                                        mod_next_rec_item['Start'] = merged_range.lower
                                    if 'End' in mod_next_rec_item:
                                        mod_next_rec_item['End'] = merged_range.upper
                                    if 'Text' in mod_next_rec_item and text:
                                        mod_next_rec_item['Text'] = text[merged_range.lower:merged_range.upper]

                                # Add unit to number range
                                if unit:
                                    if 'Resolution' in mod_next_rec_item and mod_next_rec_item['Resolution']:
                                        mod_next_rec_item['Resolution']['unit'] = unit
                                    else:
                                        mod_next_rec_item['Resolution'] = {'unit': unit}
                                # Save to modified recognized items
                                mod_rec_items[i+j+1] = mod_next_rec_item
                                # Current recognized item will not be kept
                                keep = False
                                break

                # If the the current recognized item has an empty value, remove the current item
                if has_empty_value(recognized_item):
                    keep = False

                # Remove items where 'in' recognized as Unit 'inch' from cases like "a population of 14,440 in 1787"
                if has_wrong_inch(recognized_item):
                    keep = False

                # Remove items where value is falsely recognized as a negative value
                if has_wrong_negative(recognized_item):
                    keep = False

                # Store kept items
                if keep:
                    if i not in mod_rec_items:
                        # Copy recognizers text item
                        kept_recognized_item = copy.deepcopy(recognized_item)
                        # Change offset to end + 1
                        kept_recognized_item['Start'] = kept_recognized_item['Start']
                        kept_recognized_item['End'] = kept_recognized_item['End'] + 1
                    else:
                        # Recognized item already modified
                        kept_recognized_item = mod_rec_items[i]

                    # Get type name of current recognizers text item
                    type_name = self.get_type_name(kept_recognized_item['TypeName'])
                    # Add to corresponding type
                    if type_name and type_name in self.proc_res:
                        self.proc_res[type_name].append(kept_recognized_item)

            # Sort value and time list ASC based on start offset
            values = self.proc_res[RECOGNIZER_TYPE.VALUE.label]
            if values:
                self.proc_res[RECOGNIZER_TYPE.VALUE.label] = list(sorted(values, key=lambda x: x['Start']))
            times = self.proc_res[RECOGNIZER_TYPE.TIME.label]
            if times:
                self.proc_res[RECOGNIZER_TYPE.TIME.label] = list(sorted(times, key=lambda x: x['Start']))

        return self

    def match_by_offset(self, offset: Tuple[int, int]) -> Dict[str, List[int]]:
        """
        Get in range recognizers text positions (offset of list) by offset
        :param offset: input offset
        :return: self
        """
        matched_ids = {
            RECOGNIZER_TYPE.VALUE.label: [],
            RECOGNIZER_TYPE.TIME.label: []
        }
        if self.proc_res:
            for type_name, filtered_items in self.proc_res.items():
                for index, filtered_item in enumerate(filtered_items):
                    item_offset = (filtered_item['Start'], filtered_item['End'])
                    # Check if input offset is overlapped by recognizers text item offset
                    if is_overlapped_range(offset, item_offset):
                        matched_ids[type_name].append(index)
        return matched_ids

    def get_res(self, res_type: str, res_id: int):
        """
        Get recognized result by type and id
        :param res_type: recognized result type (value or time)
        :param res_id: recognized result id (matching the index from result list)
        """
        recognized_res = None
        if res_type in self.proc_res and len(self.proc_res[res_type]) > res_id:
            recognized_res = self.proc_res[res_type][res_id]
        return recognized_res

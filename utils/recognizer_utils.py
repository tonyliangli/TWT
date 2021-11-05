import collections

from services.recognizer import Recognizer
from language.totto.baseline_preprocessing import preprocess_utils

def run_recognizer(text):
    # Run Recognizers Text and process result
    recognizer = Recognizer()
    recognized_res = recognizer.run_recognizer(text)
    recognizer.process_result(recognized_res, text)
    return recognizer

def parse_rec_res(value_rec_res, time_rec_res):
    values = collections.OrderedDict()
    if value_rec_res:
        for rec_value in value_rec_res:
            if 'Resolution' in rec_value and rec_value['Resolution']:
                    if 'value' in rec_value['Resolution'] and rec_value['Resolution']['value'] is not None:
                        value = rec_value['Resolution']['value']
                        if value not in values:
                            values[value] = []
                        # value_text = rec_value['Text']
                        # if value_text not in values[value]:
                        values[value].append(rec_value)

    times = collections.OrderedDict()
    if time_rec_res:
        for rec_time in time_rec_res:
            if 'Resolution' in rec_time and rec_time['Resolution']:
                if 'values' in rec_time['Resolution'] and rec_time['Resolution']['values']:
                    for time_value in rec_time['Resolution']['values']:
                        if 'timex' in time_value:
                            if time_value['timex'] not in times:
                                times[time_value['timex']] = []
                            # time_text = rec_time['Text']
                            # if time_text not in times[time_value['timex']]:
                            times[time_value['timex']].append(rec_time)
    return values, times


def build_sub_table(parsed_data):
    table = parsed_data["table"]

    cell_indices = parsed_data["highlighted_cells"]
    table_page_title = parsed_data["table_page_title"]
    table_section_title = parsed_data["table_section_title"]

    subtable = (
        preprocess_utils.get_highlighted_subtable(
            table=table,
            cell_indices=cell_indices,
            with_heuristic_headers=True))

    subtable_metadata_str = (
          preprocess_utils.linearize_subtable(
              subtable=subtable,
              table_page_title=table_page_title,
              table_section_title=table_section_title))

    return subtable, subtable_metadata_str


def rec_output(output_sent):
    sent_recognizer = run_recognizer(output_sent)
    output_values, output_times = parse_rec_res(sent_recognizer.proc_res['value'], sent_recognizer.proc_res['time'])
    return output_values, output_times


def rec_input(table: str, metas:list, is_sub_table=False):
    input_values, input_times = collections.OrderedDict(), collections.OrderedDict()
    if is_sub_table:
        for item in table:
            cell = item['cell']
            row_headers = item['row_headers']
            col_headers = item['col_headers']

            cell_recognized = run_recognizer(cell['value'])
            cell_values, cell_times = parse_rec_res(cell_recognized.proc_res['value'], cell_recognized.proc_res['time'])
            input_values.update(cell_values)
            input_times.update(cell_times)
    else:
        for row in table:
            for cell in row:
                cell_recognized = run_recognizer(cell['value'])
                cell_values, cell_times = parse_rec_res(cell_recognized.proc_res['value'], cell_recognized.proc_res['time'])
                input_values.update(cell_values)
                input_times.update(cell_times)

    if metas: 
        for meta in metas:
            meta_recognized = run_recognizer(meta)
            meta_values, meta_times = parse_rec_res(meta_recognized.proc_res['value'], meta_recognized.proc_res['time'])
            input_values.update(meta_values)
            input_times.update(meta_times)

    return input_values, input_times


def find_missing_info(input_values, input_times, output_values, output_times):
    missing_v_values = set(output_values.keys()).difference(set(input_values.keys()))
    missing_t_values = set(output_times.keys()).difference(set(input_times.keys()))
    
    missing_values = [rec_value for missing_v_value in missing_v_values for rec_value in output_values[missing_v_value]]
    missing_times = [rec_time for missing_t_value in missing_t_values for rec_time in output_times[missing_t_value]]
    
    # return missing_values, missing_times
    # return list(missing_v_values), missing_times
    return missing_values, missing_times

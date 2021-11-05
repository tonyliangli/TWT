import collections

from language.totto.baseline_preprocessing.preprocess_utils import (
    _add_adjusted_col_offsets,
    _get_heuristic_row_headers,
    _get_heuristic_col_headers
)


ADDITIONAL_SPECIAL_TOKENS = [
    "<page_title>",
    "</page_title>",
    "<section_title>",
    "</section_title>",
    "<meta>",
    "</meta>",
    "<table>",
    "</table>",
    "<cell>",
    "</cell>",
    "<col_header>",
    "</col_header>",
    "<row_header>",
    "</row_header>",
    "<header>",
    "</header>",
    "<extra>",
    "</extra>",
    "<value>",
    "</value>"
]


def parse_table(table, facts_coord_to_indicies):
    adjusted_table = _add_adjusted_col_offsets(table)

    adujusted_facts_coord_to_indicies = collections.OrderedDict()
    if (-1, -1) in facts_coord_to_indicies:
        adujusted_facts_coord_to_indicies[(-1, -1)] = facts_coord_to_indicies[(-1, -1)]

    parsed_cells, cell_values, row_indicies, col_indicies = [], [], [], []
    row_index_from_zero = False
    for r_index, row in enumerate(table):
        for c_index, col in enumerate(row):
            if not (r_index == 0 and col['is_header']):
                # Mark row start index
                if r_index == 0:
                    row_index_from_zero = True
                row_headers = _get_heuristic_row_headers(adjusted_table, r_index, c_index)
                col_headers = _get_heuristic_col_headers(adjusted_table, r_index, c_index)
                # parsed_cell = ','.join([header['value'] for header in col_headers + row_headers]) if col_headers + row_headers else ""
                parsed_cell = ""
                if col_headers:
                    parsed_cell = col_headers[0]['value']
                elif row_headers:
                    parsed_cell = ','.join([row_header['value'] for row_header in row_headers])
                else:
                    pass

                parsed_cell += ("|" + f"{col['value']}") if parsed_cell else f"{col['value']}"
                parsed_cells.append(parsed_cell)

                cell_values.append(col['value'])

                adjusted_row_index = r_index + 1 if row_index_from_zero else r_index
                adjusted_col_index = c_index + 1

                row_indicies.append(adjusted_row_index)
                col_indicies.append(adjusted_col_index)

                # Adujst coordinates from aligned facts
                if (r_index, c_index) in facts_coord_to_indicies:
                    adujusted_facts_coord_to_indicies[(adjusted_row_index, adjusted_col_index)] = facts_coord_to_indicies[(r_index, c_index)]

    return parsed_cells, cell_values, row_indicies, col_indicies, adujusted_facts_coord_to_indicies


def parse_linearized_table(table, facts_coord_to_indicies):
    adjusted_table = _add_adjusted_col_offsets(table)

    adujusted_facts_coord_to_indicies = collections.OrderedDict()
    if (-1, -1) in facts_coord_to_indicies:
        adujusted_facts_coord_to_indicies[(-1, -1)] = facts_coord_to_indicies[(-1, -1)]

    parsed_cells, cell_values, row_indicies, col_indicies = [], [], [], []
    row_index_from_zero = False
    for r_index, row in enumerate(table):
        for c_index, col in enumerate(row):
            if not (r_index == 0 and col['is_header']):
                # Mark row start index
                if r_index == 0:
                    row_index_from_zero = True
                row_headers = _get_heuristic_row_headers(adjusted_table, r_index, c_index)
                col_headers = _get_heuristic_col_headers(adjusted_table, r_index, c_index)

                start_cell_marker = "<cell> "
                end_cell_marker = "</cell>"

                # The value of the cell.
                parsed_cell = start_cell_marker + col['value'] + " "
                if col_headers:
                    for col_header in col_headers:
                        parsed_cell += "<header> " + col_header['value'] + " </header> "
                elif row_headers:
                    for row_header in row_headers:
                        parsed_cell += "<header> " + row_header['value'] + " </header> "
                else:
                    pass

                parsed_cell += end_cell_marker
                parsed_cells.append(parsed_cell)

                cell_values.append(col['value'])

                adjusted_row_index = r_index + 1 if row_index_from_zero else r_index
                adjusted_col_index = c_index + 1

                row_indicies.append(adjusted_row_index)
                col_indicies.append(adjusted_col_index)

                # Adujst coordinates from aligned facts
                if (r_index, c_index) in facts_coord_to_indicies:
                    adujusted_facts_coord_to_indicies[(adjusted_row_index, adjusted_col_index)] = facts_coord_to_indicies[(r_index, c_index)]

    return parsed_cells, cell_values, row_indicies, col_indicies, adujusted_facts_coord_to_indicies


def linearize_full_table(table, metas):
    """Linearize full table with localized headers and return a string."""
    table_str = ""
    for meta in metas:
        if meta:
            table_str += "<meta> " + meta + " </meta> "

    table_str += "<table> "
    adjusted_table = _add_adjusted_col_offsets(table)
    for r_index, row in enumerate(table):
        row_str = "<row> "
        for c_index, col in enumerate(row):

            row_headers = _get_heuristic_row_headers(adjusted_table, r_index,
                                                     c_index)
            col_headers = _get_heuristic_col_headers(adjusted_table, r_index,
                                                     c_index)

            start_cell_marker = "<cell> "
            end_cell_marker = "</cell> "

            # The value of the cell.
            item_str = start_cell_marker + col["value"] + " "

            # All the column headers associated with this cell.
            for col_header in col_headers:
                item_str += "<col_header> " + col_header[
                    "value"] + " </col_header> "

            # All the row headers associated with this cell.
            for row_header in row_headers:
                item_str += "<row_header> " + row_header[
                    "value"] + " </row_header> "

            item_str += end_cell_marker
            row_str += item_str

        row_str += "</row> "
        table_str += row_str

    table_str += "</table>"

    return table_str


def get_table_parent_format(table,
                            metas,
                            row_indices=None,
                            col_indices=None,
                            use_subtable=False):
    """Convert table to format required by PARENT."""
    table_parent_array = []

    # Table values.
    for row_idx, row in enumerate(table):
        for col_idx, cell in enumerate(row):
            if row_idx == 0 and cell['is_header']:
                attribute = "header"
            else:
                attribute = "cell"
            value = cell["value"].strip()
            if value:
                value = value.replace("|", "-")
                entry = "%s|||%s" % (attribute, value)
                if use_subtable:
                    if row_idx in row_indices or col_idx in col_indices:
                        table_parent_array.append(entry)
                else:
                    table_parent_array.append(entry)

    # Metas
    for i, meta in enumerate(metas):
        meta = meta.replace("|", "-")
        entry = "%s|||%s" % (f"meta {i+1}", meta)
        table_parent_array.append(entry)

    table_parent_str = "\t".join(table_parent_array)
    return table_parent_str


def get_parent_tables(table, metas, matched_facts):
    """Get tables in PARENT format for each json example."""
    # Build subtable with matched facts
    row_indicies, col_indices = set(), set()
    for i, (fact_start_idx, matched_fact) in enumerate(matched_facts.items()):
        fact_str, table_coords = matched_fact
        for table_coord in table_coords:
            if table_coord[0] != -1:
                row_indicies.add(table_coord[0])
                col_indices.add(table_coord[1])

    # Get PARENT format code for precision.
    table_prec = get_table_parent_format(
        table=table,
        metas=metas)

    # Get PARENT format code for recall.
    table_rec = get_table_parent_format(
        table=table,
        metas=metas,
        row_indices=row_indicies,
        col_indices=col_indices,
        use_subtable=True)

    return table_prec, table_rec

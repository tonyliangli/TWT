import os
import sys
import pandas as pd


def convert_table(table_path: str):
    t = pd.read_csv(table_path, delimiter="#", encoding='utf-8')
    return convert_df(t)


def convert_df(t):
    t.fillna('')
    totto_cell_format = {
         'column_span': 1,
         'is_header': False,
         'row_span': 1,
         'value': ""
    }
    header_built = False
    table_rows, header_row = [], []
    # Build header
    for r in t.itertuples(index=False):
        value_row = []
        for c_idx, c in enumerate(t.columns):
            # Build cell header
            if not header_built:
                cell_header = totto_cell_format.copy()
                cell_header["is_header"] = True
                cell_header["value"] = str(c).strip()
                header_row.append(cell_header)
            # Build cell value
            cell_value = totto_cell_format.copy()
            cell_value["value"] = str(r[c_idx]).strip()
            value_row.append(cell_value)
        # Save header row
        if not header_built:
            table_rows.append(header_row)
            header_built = True
        # Save value row
        table_rows.append(value_row)
    return table_rows


if __name__ == "__main__":
    current_dir = os.path.abspath(os.path.dirname(__file__))
    table_fact_csv_dir = f"{current_dir}/../../data/tablefact/all_csv"
    table_name = "1-1921-1.html.csv"
    convert_table(f"{table_fact_csv_dir}/{table_name}")

# data_loader.py

import pandas as pd
import openpyxl


def check_sheet_tab_colors(file_path):
    """Returns a dict of {sheet_name: color_hex} for all sheets with a tab color."""
    workbook = openpyxl.load_workbook(file_path)
    sheet_colors = {}

    for sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]
        tab_color = sheet.sheet_properties.tabColor

        if tab_color is not None:
            color_value = tab_color.rgb if tab_color.rgb else str(tab_color)
            sheet_colors[sheet_name] = color_value

    return sheet_colors


def load_colored_sheets(file_path, target_color="FF92D050"):
    """Loads only sheets whose tab color matches the target color (default is green)."""
    sheet_colors = check_sheet_tab_colors(file_path)
    green_sheets = [s for s, c in sheet_colors.items() if c == target_color]

    dfs = {}
    for sheet_name in green_sheets:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        dfs[f"{sheet_name}"] = df

    return dfs


def load_standard_excel(file_path):
    """Loads an Excel file that contains only one standard sheet."""
    df = pd.read_excel(file_path)
    return {"df_standard": df}


def load_multiple_excel_files(file_configs):
    """
    Loads multiple Excel files based on file_configs.
    file_configs = [
        { path: "...", mode: "standard"|"colored", name: "..." },
        ...
    ]
    Returns nested structure: { name: {sheetname : df} }
    """
    all_data = {}

    for config in file_configs:
        path = config["path"]
        mode = config["mode"]
        name = config["name"]

        if mode == "colored":
            data = load_colored_sheets(path)
        else:
            data = load_standard_excel(path)

        all_data[name] = data

    return all_data
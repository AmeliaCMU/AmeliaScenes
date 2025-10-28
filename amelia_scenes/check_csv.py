import os
import pandas as pd


def get_csv_files(directory):
    """Get a list of all CSV files in a directory."""
    return {file for file in os.listdir(directory) if file.endswith('.csv')}


def compare_csv_files(dir1, dir2):
    """Compare CSV files with the same name in two directories."""
    files_dir1 = get_csv_files(dir1)
    files_dir2 = get_csv_files(dir2)

    missing_in_dir2 = files_dir1 - files_dir2
    if missing_in_dir2:
        print(f"Files in {dir1} but not in {dir2}: {missing_in_dir2}")

    common_files = files_dir1.intersection(files_dir2)
    if not common_files:
        print("No common CSV files found in the directories.")
        return

    for file in common_files:
        # print(f"Comparing {file}...")
        path1 = os.path.join(dir1, file)
        path2 = os.path.join(dir2, file)

        try:
            df1 = pd.read_csv(path1)
            df2 = pd.read_csv(path2)

            # Skip files with invalid or unexpected column headers
            # Avoid ambiguous truth value on pandas Index by using `.empty` and safe string casts
            if df1.columns.empty or df2.columns.empty:
                print(f"{file}: Skipping due to empty column headers.")
                continue

            first_col1 = str(df1.columns[0]).lower()
            first_col2 = str(df2.columns[0]).lower()
            if 'version' in first_col1 or 'version' in first_col2:
                print(f"{file}: Skipping due to invalid or unexpected column headers.")
                continue

            if list(df1.columns) != list(df2.columns):
                print(f"{file}: Column mismatch.")
                print(f"Columns in {dir1}: {list(df1.columns)}")
                print(f"Columns in {dir2}: {list(df2.columns)}")
                continue

            if df1.equals(df2):
                # print(f"{file}: No differences found.")
                pass
            else:
                print(f"{file}: Differences found.")
        except Exception as e:
            print(f"Error comparing {file}: {e}")


if __name__ == "__main__":
    dir1 = "/data/aviation/Amelia-10/data/traj_data_a10v08/raw_trajectories"
    # dir1 = "/data/aviation_dev/amelia/traj_data_a10v08/raw_trajectories"
    dir2 = "/data/aviation/amelia/traj_data_a10v08/raw_trajectories"
    airport_list = ["kbos", "kdca", "kewr", "kjfk", "klax", "kmdw", "kmsy", "ksea", "ksfo", "panc"]

    if not os.path.isdir(dir1) or not os.path.isdir(dir2):
        print("One or both of the provided paths are not valid directories.")
    else:
        for airport in airport_list:
            dir1_airport = os.path.join(dir1, airport)
            dir2_airport = os.path.join(dir2, airport)

            if os.path.isdir(dir1_airport) and os.path.isdir(dir2_airport):
                print(f"Comparing CSV files for airport: {airport}")
                compare_csv_files(dir1_airport, dir2_airport)
            else:
                print(f"One or both directories for airport {airport} do not exist.")

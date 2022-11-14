"""handles input parsing."""
import pyarrow.parquet as pq
import pyarrow.csv as csv_pq
import pyarrow as pa


class Reader:
    """Parses input into required data.

    Necessary columns (reader will always look for these columns in parquet and csv):
    * videoLoc - location of video either on disc or URL
    * videoID - unique ID of each video, if not provided, ID = index

    Additional special columns:
    * caption - will be saved in separate key.txt file

    anything else - put in key.json metadata file
    """

    def __init__(self, src, meta_columns=None, url_col='videoLoc'):
        """
        Input:

        src:
            str: path to mp4 file
            str: youtube link
            str: path to txt file with multiple mp4's or youtube links
            list[str]: list with multiple mp4's or youtube links

        meta_columns:
            list[str]: columns of useful metadata to save with videos
        """
        self.columns = [url_col]
        no_dupl_temp = []
        for c in self.columns:
            if c in meta_columns:
                no_dupl_temp.append(c)
                meta_columns.remove(c)

        self.meta_columns = meta_columns if meta_columns is not None else []

        if isinstance(src, str):
            if src.endswith(".txt"):
                df = csv_pq.read_csv(
                    src, read_options=csv_pq.ReadOptions(column_names=[url_col]))
                df = df.add_column(
                    0, "videoID", [list(range(df.num_rows))])  # add ID's
            elif src.endswith(".csv"):
                df = csv_pq.read_csv(src)
                df = df.add_column(0, "videoID", [list(range(df.num_rows))])
            elif src.endswith(".parquet"):
                with open(src, "rb") as f:
                    columns_to_read = self.columns + meta_columns
                    df = pq.read_table(f, columns=columns_to_read)
                    df = df.add_column(
                        0, "videoID", [list(range(df.num_rows))])
            else:  # singular video (mp4 or link)
                src = [src]
        if isinstance(src, list):
            df = pa.Table.from_arrays([src], names=[url_col])
            df = df.add_column(
                0, "videoID", [list(range(df.num_rows))])  # add ID's

        for c in no_dupl_temp:
            self.meta_columns.append(c)
        self.df = df

    def get_data(self):
        vids = self.df[self.columns[0]].to_pylist()
        ids = self.df["videoID"]
        meta = dict(  # pylint: disable=consider-using-dict-comprehension
            [(meta, self.df[meta]) for meta in self.meta_columns]
        )
        return vids, ids, meta

import os
import pandas as pd

class DAICWOZLoader:
    def __init__(self, raw_dir="data/raw", processed_dir="data/processed"):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)

    def load_split(self, split="train"):
        """
        Loads train/dev/test splits.
        These files already contain labels (PHQ, depression).
        """
        path = os.path.join(self.raw_dir, f"{split}_split_Depression_AVEC2017.csv")
        df = pd.read_csv(path)

        # Normalize columns
        df.columns = [c.lower() for c in df.columns]
        df["split"] = split

        return df

    def build_metadata(self):
        """
        Creates metadata with ONLY participant_id + labels + split.
        With features
        """
        print("Building metadata (NO audio/text)...")

        # Load labeled splits
        train = self.load_split("train")
        dev = self.load_split("dev")
        test = self.load_split("test")

        df = pd.concat([train, dev, test], ignore_index=True)

        # Add NULL placeholders for now
        df["audio_path"] = None
        df["transcript_path"] = None

        feature_paths = {
        "covarep_path": [],
        "formant_path": [],
        "au_path": [],
        "gaze_path": [],
        "pose_path": [],
        "landmarks_path": [],
        }

        for pid in df["participant_id"]:
            feats = self.find_feature_files(pid)

            feature_paths["covarep_path"].append(feats["covarep"])
            feature_paths["formant_path"].append(feats["formant"])
            feature_paths["au_path"].append(feats["au"])
            feature_paths["gaze_path"].append(feats["gaze"])
            feature_paths["pose_path"].append(feats["pose"])
            feature_paths["landmarks_path"].append(feats["landmarks"])

        # Append to dataframe
        for k, v in feature_paths.items():
            df[k] = v

        # Save metadata
        out_path = os.path.join(self.processed_dir, "metadata.csv")
        df.to_csv(out_path, index=False)

        print(f"Saved metadata → {out_path}")
        return df
    
    def find_feature_files(self, pid):
        """Return dict of paths to feature files for a participant."""
        # Try both {pid} and {pid}_P directory names
        session_dir = os.path.join(self.raw_dir, str(pid))
        if not os.path.isdir(session_dir):
            session_dir = os.path.join(self.raw_dir, f"{pid}_P")
        if not os.path.isdir(session_dir):
            return {
                "covarep": None,
                "formant": None,
                "au": None,
                "gaze": None,
                "pose": None,
                "landmarks": None,
            }

        feats = {
            "covarep": None,
            "formant": None,
            "au": None,
            "gaze": None,
            "pose": None,
            "landmarks": None,
        }

        # Look for COVAREP and FORMANT in subdirectories first
        covarep_dir = os.path.join(session_dir, "COVAREP")
        if os.path.isdir(covarep_dir):
            for f in os.listdir(covarep_dir):
                if f.endswith(".csv"):
                    feats["covarep"] = os.path.join(covarep_dir, f)
                    break
        else:
            # Look for COVAREP files directly in session directory
            for f in os.listdir(session_dir):
                if "covarep" in f.lower() and f.endswith(".csv"):
                    feats["covarep"] = os.path.join(session_dir, f)
                    break

        formant_dir = os.path.join(session_dir, "FORMANT")
        if os.path.isdir(formant_dir):
            for f in os.listdir(formant_dir):
                if f.endswith(".csv"):
                    feats["formant"] = os.path.join(formant_dir, f)
                    break
        else:
            # Look for FORMANT files directly in session directory
            for f in os.listdir(session_dir):
                if "formant" in f.lower() and f.endswith(".csv"):
                    feats["formant"] = os.path.join(session_dir, f)
                    break

        # Look for OpenFace files (CLNF format: *_CLNF_AUs.txt, *_CLNF_gaze.txt, etc.)
        # First try openface subdirectory
        openface_dir = os.path.join(session_dir, "openface")
        if os.path.isdir(openface_dir):
            for f in os.listdir(openface_dir):
                name = f.lower()
                fpath = os.path.join(openface_dir, f)
                if "au" in name and (name.endswith(".txt") or name.endswith(".csv")):
                    feats["au"] = fpath
                elif "gaze" in name and (name.endswith(".txt") or name.endswith(".csv")):
                    feats["gaze"] = fpath
                elif "pose" in name and (name.endswith(".txt") or name.endswith(".csv")):
                    feats["pose"] = fpath
                elif ("landmark" in name or "points" in name) and (name.endswith(".txt") or name.endswith(".csv")):
                    feats["landmarks"] = fpath
        else:
            # Look for CLNF files directly in session directory
            for f in os.listdir(session_dir):
                name = f.lower()
                fpath = os.path.join(session_dir, f)
                if "clnf_aus" in name or ("au" in name and name.endswith(".txt")):
                    feats["au"] = fpath
                elif "clnf_gaze" in name or ("gaze" in name and name.endswith(".txt")):
                    feats["gaze"] = fpath
                elif "clnf_pose" in name or ("pose" in name and name.endswith(".txt")):
                    feats["pose"] = fpath
                elif ("landmark" in name or "points" in name) and name.endswith(".txt"):
                    feats["landmarks"] = fpath

        return feats


# import os
# import pandas as pd
# import glob

# class DAICWOZLoader:
#     def __init__(self, raw_dir="data/raw", processed_dir="data/processed"):
#         self.raw_dir = raw_dir
#         self.processed_dir = processed_dir
#         os.makedirs(processed_dir, exist_ok=True)

#     def load_split(self, split="train"):
#         """
#         Loads train/dev/test splits.
#         These files already contain labels.
#         """
#         path = os.path.join(self.raw_dir, f"{split}_split_Depression_AVEC2017.csv")
#         df = pd.read_csv(path)

#         df.columns = [c.lower() for c in df.columns]
#         df["split"] = split
#         return df

#     def find_session_dir(self, pid):
#         """
#         Finds the folder for a participant.
#         Supports:
#             303/
#             303_P/
#             303_*/(anything)
#         """
#         pattern1 = os.path.join(self.raw_dir, str(pid))
#         pattern2 = os.path.join(self.raw_dir, f"{pid}_*")

#         # Exact match
#         if os.path.isdir(pattern1):
#             return pattern1

#         # Pattern match (like 303_P)
#         matches = glob.glob(pattern2)
#         if len(matches) > 0:
#             return matches[0]

#         return None

#     def build_metadata(self):
#         """
#         Creates metadata with paths + labels.
#         """
#         print("Building metadata...")

#         train = self.load_split("train")
#         dev = self.load_split("dev")
#         test = self.load_split("test")
#         df = pd.concat([train, dev, test], ignore_index=True)

#         audio_paths = []
#         transcript_paths = []

#         for pid in df["participant_id"]:
#             session_dir = self.find_session_dir(pid)

#             wav_file = None
#             transcript_file = None

#             if session_dir and os.path.isdir(session_dir):
#                 for f in os.listdir(session_dir):
#                     fp = os.path.join(session_dir, f)

#                     if f.lower().endswith(".wav"):
#                         wav_file = fp

#                     if "transcript" in f.lower():
#                         transcript_file = fp

#             audio_paths.append(wav_file)
#             transcript_paths.append(transcript_file)

#         df["audio_path"] = audio_paths
#         df["transcript_path"] = transcript_paths

#         out_path = os.path.join(self.processed_dir, "metadata.csv")
#         df.to_csv(out_path, index=False)

#         print(f"Saved metadata → {out_path}")
#         return df

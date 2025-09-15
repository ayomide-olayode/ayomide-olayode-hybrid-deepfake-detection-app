import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------- CONFIG ----------
fakeav_root = "./data/FakeAVCeleb_v1.2"  # change to your dataset root
metadata_file = os.path.join(fakeav_root, "meta_data.csv")
data_root = "./data"
test_size = 0.2
random_state = 42
video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.mpg', '.mpeg'}
# ----------------------------

df = pd.read_csv(metadata_file)
print("Columns:", df.columns.tolist())

# detect which column holds the folder (unnamed) and which holds the stem
folder_col = next((c for c in df.columns if c.lower().startswith('unnamed')), None)
stem_col = 'path'  # per your description

if folder_col is None:
    raise SystemExit("Could not find an 'Unnamed' folder column. Check CSV.")

# keep only categories A (real) and C (fake)
df = df[df['category'].isin(['A', 'C'])].copy()

# face identity
def get_face_identity(row):
    return row['source'] if row['category'] == 'A' else row['target1']

df['face_identity'] = df.apply(get_face_identity, axis=1)

# split identities (80/20)
identities = df['face_identity'].unique()
train_ids, test_ids = train_test_split(identities, test_size=test_size, random_state=random_state)
split_map = {id_: ('train' if id_ in train_ids else 'test') for id_ in identities}

# make dirs
for split in ('train', 'test'):
    for label in ('real', 'fake'):
        os.makedirs(os.path.join(data_root, split, label), exist_ok=True)

# helpers
def candidate_dirs(folder_value):
    """Return plausible absolute dirs to try based on folder_value."""
    v = str(folder_value).strip()
    cands = []
    if os.path.isabs(v):
        cands.append(v)
    cands.append(os.path.join(fakeav_root, v))
    # If path contains 'FakeAVCeleb' strip the prefix and join
    if 'FakeAVCeleb' in v:
        after = v.split('FakeAVCeleb', 1)[1].lstrip('/\\')
        if after:
            cands.append(os.path.join(fakeav_root, after))
    parent = os.path.dirname(fakeav_root)
    cands.append(os.path.join(parent, v))
    return [os.path.normpath(x) for x in cands if x]

def find_files_by_stem(dirpath, stem):
    """Find files in dir whose basename (without ext) equals stem.
       If stem already contains an extension, prefer exact match."""
    stem = str(stem).strip()
    found = []
    if '.' in stem:
        # user already provided extension
        candidate = os.path.join(dirpath, stem)
        if os.path.isfile(candidate):
            found.append(candidate)
        return found

    try:
        for name in os.listdir(dirpath):
            full = os.path.join(dirpath, name)
            if not os.path.isfile(full):
                continue
            base, ext = os.path.splitext(name)
            if base == stem and ext.lower() in video_exts:
                found.append(full)
    except Exception:
        pass
    return found

copied_srcs = set()
counts = {'train_real':0, 'train_fake':0, 'test_real':0, 'test_fake':0,
          'txt_copied':0, 'txt_missing':0, 'video_missing':0, 'rows_unresolved':0}

for idx, row in df.iterrows():
    folder_val = row.get(folder_col, '')
    stem_val = row.get(stem_col, '')

    if pd.isna(folder_val) or pd.isna(stem_val) or str(stem_val).strip() == '':
        counts['rows_unresolved'] += 1
        print(f"[WARN] missing folder or stem at row {idx}")
        continue

    # find the directory that exists
    dir_found = None
    for cand in candidate_dirs(folder_val):
        if os.path.isdir(cand):
            dir_found = cand
            break
    if dir_found is None:
        print(f"[WARN] no directory found for folder '{folder_val}' (row {idx})")
        counts['video_missing'] += 1
        continue

    # the stem column might contain multiple stems separated by comma/semicolon; handle that
    stems = [s.strip() for s in str(stem_val).replace(';', ',').split(',') if s.strip()]

    face_id = row['face_identity']
    split = split_map.get(face_id, 'train')
    label = 'real' if row['category'] == 'A' else 'fake'
    dest_dir = os.path.join(data_root, split, label)

    for stem in stems:
        matches = find_files_by_stem(dir_found, stem)
        if not matches:
            # as a fallback, try joining dir + stem + common video ext (.mp4) to be tolerant
            fallback = os.path.join(dir_found, stem + '.mp4')
            if os.path.isfile(fallback):
                matches = [fallback]

        if not matches:
            print(f"[MISSING] no video for stem '{stem}' in {dir_found}")
            counts['video_missing'] += 1
            continue

        for orig_video in matches:
            real_src = os.path.realpath(orig_video)
            if real_src in copied_srcs:
                continue
            dest_video = os.path.join(dest_dir, os.path.basename(orig_video))
            try:
                shutil.copy2(orig_video, dest_video)
                copied_srcs.add(real_src)
                counts[f"{split}_{label}"] += 1
                print(f"Copied video {orig_video} -> {dest_video}")
            except Exception as e:
                print(f"[ERROR] copying {orig_video}: {e}")
                continue

            # copy .txt sidecar with same stem
            stem_base = os.path.splitext(os.path.basename(orig_video))[0]
            txt_src = os.path.join(dir_found, stem_base + '.txt')
            txt_dest = os.path.join(dest_dir, stem_base + '.txt')
            if os.path.isfile(txt_src):
                try:
                    shutil.copy2(txt_src, txt_dest)
                    counts['txt_copied'] += 1
                    print(f"Copied txt {txt_src} -> {txt_dest}")
                except Exception as e:
                    print(f"[ERROR] copying txt {txt_src}: {e}")
            else:
                counts['txt_missing'] += 1
                print(f"[NO TXT] expected {txt_src}")

# save split map
split_df = pd.DataFrame(list(split_map.items()), columns=['face_identity', 'split'])
split_df.to_csv(os.path.join(data_root, "identity_splits.csv"), index=False)

print("Done.")
print(counts)

import base64, hashlib, io, json, os, shutil, subprocess, sys, zipfile
STEP = "babel_llama"  # the whole run: model check, number-format check, known-answer test, tokenizer comparison, then the experiments
HOURS = 10.5  # jobs stop cleanly after this and everything is packaged; run again with this notebook's output attached to continue
SRC = "/tmp/babel_src"

blob = base64.b64decode("".join(PAYLOAD.split()))  # PAYLOAD comes from the cell above
sha = "bundle-" + hashlib.sha256(blob).hexdigest()[:12]
assert hashlib.sha256(blob).hexdigest() == PAYLOAD_SHA256, "the packed code is damaged: upload the notebook file again"
shutil.rmtree(SRC, ignore_errors=True)
zipfile.ZipFile(io.BytesIO(blob)).extractall(SRC)
print("code packed in this notebook: " + sha)
env = dict(os.environ, BABEL_STEP=STEP, HOURS=str(HOURS), PYTHONUNBUFFERED="1", BABEL_SRC=SRC, BABEL_SRC_SHA=sha)
p = subprocess.Popen([sys.executable, os.path.join(SRC, "kaggle", "driver.py")], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    print(line, end="")
p.wait()
print("driver exit code:", p.returncode, "(0 = everything finished)")
f = "/kaggle/working/results/RUN_SUMMARY.json"
print(json.dumps(json.load(open(f)), indent=1) if os.path.exists(f) else "no summary (see the messages above)")
try:  # show the graphs under the cell as well as saving them in the Output
    import glob
    from IPython.display import Image, display
    for g in sorted(glob.glob("/kaggle/working/results/figures_kaggle/*.png")):
        print(g)
        display(Image(filename=g))
except Exception as e:
    print("graphs not shown here (" + type(e).__name__ + "); they are in the Output")

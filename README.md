# rs_d435

Repo for stream and record data from Intel RealSense D435.

## Examples
For test image show:
```bash
python3 rs_d435.py -t
```

For data stream enable (color + depth):
```bash
python3 rs_d435.py -s
```

For data stream record (color + depth + distance):
```bash
python3 rs_d435.py -s -r
```

| Flag          | Description                  |
| ------------- | ---------------------------- |
| -t            | Test demo                    |
| -s            | Enable stream RealSense D435 |
| -r            | Enable stream record         |
| -sh           | Enable stream window show    |

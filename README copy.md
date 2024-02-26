# Budget-Constrained Tool Learning with Planning

This is the code of the paper "Budget-Constrained Tool Learning with Planning".

## Requirements

See [the requirements of ToolBench](https://github.com/OpenBMB/ToolBench/blob/master/requirements.txt).

## Usage

### Download the Data and the Model

1. Download the data file `data.zip` from [Google Drive](https://drive.google.com/drive/folders/1yBUQ732mPu-KclJnuQELEhtKakdXFc3J).

2. Download the data and model file `data_and_model.zip` from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/1cbe2cf755e147538ff9/?dl=1).

3. Unzip the downloaded zip files and place the data and model files like this:
```
BTP
 |-data
    |- ...
 |-data2
    |- ...
 |-trained_model
    |- ...
 |-...
```

### Modify the Code

1. Modify the script file `infer.sh`: 

Replace `$YOUR_TOOLBENCH_KEY` with your ToolBench key. Replace `$YOUR_OPENAI_KEY` with your OpenAI key.

2. Modify the Python file `toolbench/inference/Downstream_tasks/rapidapi.py`:

Replace `YOUR_ABSOLUTE_PATH_OF_TRAINED_MODEL` (line 552) with the absolute path of the directory `trained_model` (which is unzipped from `data_and_model.zip`).

3. Prepare an API pool file, which is a JSON file like below:
```
[
    {
        "username": "your_user_name",
        "passwd": "your_password",
        "api_key": "your_openai_key",
        "organization": "your_organization"
    },
    ...
]
```

4. Modify the script file `toolbench/tooleval/run_pass_rate.sh`:

Replace `$YOUR_API_POOL_FILE` to the absolute path of the API pool file you prepared as above.

### Prepare the Plan

```
bash prepare.sh
```

### Infer with the Plan

```
bash infer.sh $SUBSET
```

Note: `$SUBSET` is one of the strings below:
`G1_instruction`, `G1_tool`, `G1_category`, `G2_instruction`, `G2_category`, `G3_instruction`.

### Evaluate the Results

```
cd toolbench/tooleval
bash eval.sh $SUBSET
```
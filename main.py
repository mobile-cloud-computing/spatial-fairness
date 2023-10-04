
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from application import calculate_fairness_metrics  # process_uploaded_file, 


app = FastAPI()


@app.post('/upload')
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail='No selected file')


    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail='Invalid file format')


    try:
        file_path = 'temp.csv'
        with open(file_path, 'wb') as f:
            f.write(file.file.read())


        # Call the processing function from application.py
        # df1 = process_uploaded_file(file_path)


        # Calculate fairness metrics
        fairness_summary= calculate_fairness_metrics(file_path)
        # fairness_for_age = calculate_fairness_metrics(df1, 'Age')


        # Format the results as JSON
        results = {
            'message': 'File uploaded and processed successfully',
            'fairness_summary': fairness_summary
        }


        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8083)

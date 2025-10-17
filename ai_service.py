import json
from flask import Flask, request, jsonify
from flask_cors import CORS 

app = Flask(__name__)
CORS(app) 

# --- Core Analysis Logic (Using Placeholder for Testing) ---
def perform_analysis(data):
    """
    Performs a simple, deterministic analysis for integration testing.
    """
    resume_content = data.get('text_content')
    
    if not resume_content:
        return {
            "error": "Missing 'text_content' key in received JSON payload from Java.",
            "score": 0
        }

    try:
        content_length = len(resume_content)
        
        # Dynamic Placeholder Logic: Score and summary change based on input length
        if content_length < 50:
            score = 65
            summary = f"Placeholder report: Text is too short ({content_length} chars). Score reflects basic content."
            skills = ["Basic Communication", "Data Entry"]
        else:
            score = 95 
            summary = f"Integration successful! Processed {content_length} characters of text. This report is a placeholder demonstrating full system functionality."
            skills = ["Integration Testing", "Microservice Architecture", "API Debugging"]
            
        analysis_report = {
            "score": score,
            "skills": skills,
            "summary": summary,
            "status": "Success"
        }
        return analysis_report

    except Exception as e:
        return {
            "error": f"Internal Python Analysis Failed: {type(e).__name__}: {str(e)}",
            "score": 5
        }


# --- Flask Route (Unchanged) ---
@app.route('/analyze-cv', methods=['POST'])
def analyze_cv():
    try:
        request_data = request.get_json()
        
        if request_data is None:
             return jsonify({
                "error": "Failed to parse JSON. Check Java's Content-Type header.",
                "score": 0
            }), 400

        analysis_report = perform_analysis(request_data)
        
        # Always return 200 OK if the Python service successfully processed the request
        return jsonify(analysis_report), 200

    except Exception as e:
        return jsonify({
            "error": f"Unhandled Flask server exception: {str(e)}", 
            "details": str(e)
        }), 500


if __name__ == '__main__':
    # Running on 0.0.0.0 makes it accessible to the Java app on localhost
    app.run(debug=False, host='0.0.0.0', port=5000)
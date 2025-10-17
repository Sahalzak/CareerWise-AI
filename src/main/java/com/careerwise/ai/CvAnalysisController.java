package com.careerwise.ai;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.Map; // Used to automatically map the incoming JSON body

@RestController
@RequestMapping("/") 
public class CvAnalysisController {

    private final AnalysisService analysisService;

    public CvAnalysisController(AnalysisService analysisService) {
        this.analysisService = analysisService;
    }

    // Now accepts a JSON body and extracts the text
    @PostMapping("/api/analyze")
    // @RequestBody automatically maps the incoming JSON body to a Map
    public ResponseEntity<String> analyzeCv(@RequestBody Map<String, String> payload) { 
        
        // The frontend sends data with the key "resume_content"
        String resumeText = payload.get("resume_content");

        if (resumeText == null || resumeText.trim().isEmpty()) {
            return ResponseEntity.badRequest().body("Resume content is missing or empty.");
        }
        
        try {
            // Call the service with the raw text (renamed method below)
            String analysisResult = analysisService.analyzeText(resumeText);

            return ResponseEntity.ok(analysisResult);
        } catch (Exception e) {
            return ResponseEntity.status(500).body("Analysis failed: " + e.getMessage());
        }
    }
    
    // NOTE: The @GetMapping("/") method remains in ViewController.java
}
package com.careerwise.ai;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType; 
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

@Service
public class AnalysisService {

    @Value("${python.api.url}")
    private String pythonApiUrl;

    private final RestTemplate restTemplate = new RestTemplate();

    // Renamed method to reflect text input
    public String analyzeText(String resumeText) {
        // --- 1. SET HTTP HEADERS (CRUCIAL FIX) ---
        HttpHeaders headers = new HttpHeaders();
        // Tells the Python server the body is JSON
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.setAccept(Collections.singletonList(MediaType.APPLICATION_JSON));

        // --- 2. PREPARE THE JSON PAYLOAD (Key updated to match Python's expectation) ---
        Map<String, String> requestPayload = new HashMap<>();
        requestPayload.put("text_content", resumeText); // New key name for Python

        // --- 3. CREATE THE HTTP ENTITY ---
        HttpEntity<Map<String, String>> entity = new HttpEntity<>(requestPayload, headers);

        try {
            // --- 4. SEND THE REQUEST ---
            ResponseEntity<String> response = restTemplate.exchange(
                pythonApiUrl + "/analyze-cv",
                HttpMethod.POST,
                entity,
                String.class
            );

            if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
                return response.getBody();
            } else {
                return "ERROR: Python API returned status: " + response.getStatusCode().value() + ". Body: " + response.getBody();
            }

        } catch (Exception e) {
            return "ERROR: Failed to connect or receive response from Python service: " + e.getMessage();
        }
    }
}
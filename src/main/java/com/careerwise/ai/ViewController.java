package com.careerwise.ai;

import org.springframework.stereotype.Controller; // <-- IMPORTANT: Use @Controller
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller // <-- Tells Spring to use the View Resolver
@RequestMapping("/")
public class ViewController {

    // Serves the static index.html file from the root
    @GetMapping("/")
    public String index() {
        // This is correctly interpreted as a view path by @Controller
        return "forward:/index.html"; 
    }
}
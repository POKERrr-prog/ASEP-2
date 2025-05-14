// DOM Ready
document.addEventListener("DOMContentLoaded", () => {
    const authModal = document.getElementById("authModal");
    const appContainer = document.getElementById("appContainer");
    const authForm = document.getElementById("authForm");
    const navLinks = document.querySelectorAll(".nav-link");
    const pages = document.querySelectorAll(".page");
    const usernameDisplay = document.querySelector(".username");
  
    authForm.addEventListener("submit", function (e) {
      e.preventDefault();
      const username = document.getElementById("username").value.trim();
      if (username) {
        authModal.style.display = "none";
        appContainer.style.display = "flex";
        usernameDisplay.textContent = `Welcome, ${username}`;
      }
    });
    
  
    // Handle nav link clicks
    navLinks.forEach(link => {
      link.addEventListener("click", (e) => {
        e.preventDefault();
  
        // Toggle active nav item
        navLinks.forEach(l => l.classList.remove("active"));
        link.classList.add("active");
  
        // Show relevant page
        const page = link.getAttribute("data-page");
        pages.forEach(p => {
          p.classList.remove("active");
          if (p.id === page + "Page") {
            p.classList.add("active");
          }
        });
      });
    });
  
    // Webcam functionality (Search Page)
    const video = document.getElementById("webcam");
    const startWebcamBtn = document.getElementById("startWebcam");
    const captureBtn = document.getElementById("captureBtn");
    const letterDisplay = document.getElementById("letterDisplay");
    const resetBtn = document.getElementById("resetBtn");
    const searchBtn = document.getElementById("searchBtn");
    const captureHistory = document.querySelector(".captures-grid");
  
    if (startWebcamBtn) {
      startWebcamBtn.addEventListener("click", async () => {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ video: true });
          video.srcObject = stream;
        } catch (err) {
          alert("Unable to access camera.");
          console.error(err);
        }
      });
    }
  
    if (captureBtn) {
      captureBtn.addEventListener("click", () => {
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  
        const img = document.createElement("img");
        img.src = canvas.toDataURL("image/png");
        img.classList.add("capture-img");
        captureHistory.appendChild(img);
  
        // Send the captured image to the Flask backend
        const base64Image = canvas.toDataURL("image/png");
        fetch("http://127.0.0.1:5000/predict", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ image: base64Image }),
        })
          .then((response) => response.json())
          .then((data) => {
            if (data.prediction) {
              letterDisplay.textContent += data.prediction; // Append the predicted letter
            } else {
              alert("Prediction failed: " + data.error);
            }
          })
          .catch((error) => {
            console.error("Error:", error);
            alert("An error occurred while predicting.");
          });
      });
    }
  
    if (resetBtn) {
      resetBtn.addEventListener("click", () => {
        letterDisplay.textContent = "";
        captureHistory.innerHTML = "";
      });
    }
  
    if (searchBtn) {
      searchBtn.addEventListener("click", () => {
        const query = letterDisplay.textContent.trim();
        if (query) {
          window.open(`https://www.google.com/search?q=${query}`, "_blank");
        }
      });
    }
  
    // Logout
    document.querySelector(".btn-logout").addEventListener("click", () => {
      appContainer.style.display = "none";
      authModal.style.display = "flex";
    });
  });

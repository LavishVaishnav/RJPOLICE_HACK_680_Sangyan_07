// Initialize the carousel when the DOM is ready
document.addEventListener("DOMContentLoaded", function() {
  // Get the carousel element
  let myCarousel = document.getElementById("carouselExampleIndicators");

  // Create a new Bootstrap Carousel instance
  let carousel = new bootstrap.Carousel(myCarousel, {
    interval: 2000, // Set the interval for automatic sliding (in milliseconds)
    wrap: true,     // Enable looping of slides
    keyboard: true   // Enable keyboard navigation
  });
});

// Add this part to your existing code
// Elements
let btn = document.querySelector(".submit-btn");
let usernameInput = document.querySelector(".login--username");
let passwordInput = document.getElementById("exampleInputPassword1");
let wrongPassDiv = document.getElementById("wrong-pass-text");

// Define valid credentials (replace with your actual credentials)
const validUsername = "user";
const validPassword = "1111";

// Event listener for the login button
btn.addEventListener("click", function () {
  // Check if the entered credentials are valid
  if (usernameInput.value === validUsername && passwordInput.value === validPassword) {
    // Redirect to the home page or perform other actions on successful login
    window.location.href = "index.html";
  } else {
    // Display an error message for invalid credentials
    wrongPassDiv.classList.remove("opacity-0");
  }
});

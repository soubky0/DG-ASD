// Function to handle file upload
function handleFileUpload() {
  var formData = new FormData(document.getElementById("uploadForm"));
  document.getElementById("file").disabled = true;

  var xhr = new XMLHttpRequest();
  xhr.open("POST", "/upload", true);

  xhr.onload = function () {
    if (xhr.status === 200) {
      var response = JSON.parse(xhr.responseText);
      displayMessage("message", response.message, "success");
    } else {
      var response = JSON.parse(xhr.responseText);
      console.log(response);
      displayMessage("message", response.message, "danger");
      document.getElementById("file").disabled = false;
    }
  };

  xhr.send(formData);
}

// Function to handle form submission
function handleSubmit() {
  var formData = new FormData(document.getElementById("testForm"));

  var xhr = new XMLHttpRequest();
  xhr.open("POST", "/test", true);

  xhr.onload = function () {
    if (xhr.status === 200) {
      var response = JSON.parse(xhr.responseText);
      displayMessage("message2", response.message, "success");
    } else {
      var response = xhr.responseText;
      displayMessage("message2", response, "danger");
    }
  };

  xhr.send(formData);
}

// Function to display messages
function displayMessage(elementId, message, type) {
  document.getElementById(elementId).innerHTML =
    '<div class="alert alert-' + type + '" role="alert">' + message + "</div>";
}

// Event Listeners
document
  .getElementById("uploadForm")
  .addEventListener("change", handleFileUpload);
document.getElementById("submitBtn").addEventListener("click", handleSubmit);

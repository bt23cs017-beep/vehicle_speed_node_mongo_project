const input = document.getElementById("videoInput");
const selectedFile = document.getElementById("selectedFile");

input.addEventListener("change", () => {
  selectedFile.textContent = input.files.length
    ? `Selected: ${input.files[0].name}`
    : "No video selected";
});

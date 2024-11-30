document.getElementById('fileInput').addEventListener('change', handleFileSelect);
document.getElementById('dropZone').addEventListener('click', () => document.getElementById('fileInput').click());
document.getElementById('dropZone').addEventListener('dragover', (event) => {
    event.preventDefault();
    event.stopPropagation();
});
document.getElementById('dropZone').addEventListener('drop', (event) => {
    event.preventDefault();
    event.stopPropagation();
    handleFileSelect(event);
});

function handleFileSelect(event) {
    let file;
    if (event.dataTransfer) {
        file = event.dataTransfer.files[0];
    } else {
        file = event.target.files[0];
    }
    if (file && file.type.startsWith('image/')) {
        let reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('preview').src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
}

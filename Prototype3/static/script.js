document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();
    
    const fileInput = document.getElementById('imageInput');
    const formData = new FormData();
    
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select an image file.');
        return;
    }
    
    formData.append('file', file);
    
    // List of model endpoints
    const modelEndpoints = [
        'retinexnet_model',
        'low_light_model',
        'llflow_model',
        'deep_upe_model',
        'yolov5500'
    ];
    
    // Fetch and display images for each model
    modelEndpoints.forEach(model => {
        fetch(`/upload?model=${model}`, {
            method: 'POST',
            body: formData
        })
        .then(response => response.blob())
        .then(blob => {
            const url = URL.createObjectURL(blob);
            document.getElementById(`${model}Image`).src = url;
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred. Please try again.');
        });
    });
    
    // Display the original image
    document.getElementById('originalImage').src = URL.createObjectURL(file);
});

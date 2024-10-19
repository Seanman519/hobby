document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const formData = new FormData();
    const fileInput = document.getElementById('pdfInput');
    formData.append('pdf', fileInput.files[0]);

    fetch('/upload_pdf', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('pdfText').innerText = data.extracted_text;
    });
});

document.getElementById('trainBtn').addEventListener('click', function() {
    fetch('/train_model', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('trainStatus').innerText = data.message;
    });
});

document.getElementById('chatBtn').addEventListener('click', function() {
    const userInput = document.getElementById('userInput').value;

    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_input: userInput })
    })
    .then(response => response.json())
    .then(data => {
        const chatOutput = document.getElementById('chatOutput');
        const newChat = document.createElement('p');
        newChat.innerText = `AI: ${data.response}`;
        chatOutput.appendChild(newChat);
    });
});

document.getElementById('searchForm').addEventListener('submit', function (event) {
    event.preventDefault();

    // Get form input
    const premise = document.getElementById('premise').value.trim();
    const hypothesis = document.getElementById('hypothesis').value.trim();
    let isValid = true;

    // Clear previous error messages
    document.getElementById('premiseTextError').textContent = '';
    document.getElementById('hypothesisTextError').textContent = '';

    // Validate search text input
    if (!premise || premise.length === 0) {
        document.getElementById('premiseTextError').textContent = 'Please enter a valid premise text.';
        isValid = false;
    }

    if (!hypothesis || hypothesis.length === 0) {
        document.getElementById('hypothesisTextError').textContent = 'Please enter a valid hypothesis text.';
        isValid = false;
    }

    if (isValid) {
        predictRelevantContent(premise, hypothesis);
    }
});

function predictRelevantContent(premise, hypothesis) {
    const apiUrl = `/predict?premise=${encodeURIComponent(premise)}&hypothesis=${encodeURIComponent(hypothesis)}`;

    fetch(apiUrl)
        .then(response => response.json())
        .then(data => {
            if (data) {
                const resultContainer = document.getElementById('resultContainer');
                const searchResultElement = document.getElementById('searchResult');

                searchResultElement.innerHTML = '';
                searchResultElement.innerHTML = `${data}`;
                
                resultContainer.style.display = 'block';
            } else {
                console.error('Unexpected API response format:', data);
            }
        }).catch(error => console.error('Error:', error));
}
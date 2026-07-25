# GradeMatch

A photo-grading web app that copies the editing style of any reference photo onto an unedited picture, even when the two images share nothing in common.

## Overview

GradeMatch allows you to upload a photo whose look you love plus your own shot, and the app transfers the editing style from the reference image to your photo. Perfect for achieving consistent aesthetics across your photo collection.

## Features

- 📸 Upload reference and target photos
- 🎨 Automatic style transfer between images
- 🔧 Photo editing style matching
- 🌐 Web-based interface

## Tech Stack

- **Backend**: Python (55.9%)
- **Frontend**: HTML (44%)
- **Containerization**: Docker (0.1%)

## Getting Started

### Prerequisites

- Python 3.x
- Docker (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/gkarana/Grade-match-recipe.git
cd Grade-match-recipe
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
python app.py
```

The web app will be available at `http://localhost:5000` (or your configured port).

### Running with Docker

```bash
docker build -t gradematch .
docker run -p 5000:5000 gradematch
```

## Usage

1. Open the web application in your browser
2. Upload a reference photo (the one with the editing style you want)
3. Upload your own photo (the one you want to edit)
4. Click the "Match Grade" button
5. Download your edited photo with the matched style

## Project Structure

```
Grade-match-recipe/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── templates/            # HTML templates
└── static/               # Static assets
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to suggest improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or feedback, please reach out to the repository owner.

---

**Made with ❤️ by gkarana**

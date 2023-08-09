import streamlit as st
from utils.utils import decodeImage
from predict import dogcat

class ClientApp:
    def __init__(self, filename):
        self.filename = filename
        self.classifier = dogcat(self.filename)





def main():
    st.title("Dog and Cat Prediction App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    if st.button('Predict'):
        if uploaded_file is not None:
            with open("inputImage.jpg", "wb") as f:
                f.write(uploaded_file.getvalue())

            clApp = ClientApp("inputImage.jpg")
            result = clApp.classifier.predictiondogcat()

            # Extracting the prediction from the result
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                prediction = result[0].get('image', '')
            else:
                prediction = "Unable to extract prediction from the model's response."

            # Display the prediction
            st.write(prediction.capitalize())
        else:
            st.write("Please upload an image first.")

if __name__ == "__main__":
    main()
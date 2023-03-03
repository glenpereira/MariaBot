import React, { useState } from "react";
import axios from "axios";
const MODEL_API_URL = "http://127.0.0.1:5000/text";

function App() {
  const [inputText, setInputText] = useState("");

  async function handleSubmit(e) {
    e.preventDefault();
    const text = {
      text: inputText,
    };
    await axios
      .post(MODEL_API_URL, text, { responseType: "blob" })
      .then((res) => {
        // create file link in browser's memory
        const url = window.URL.createObjectURL(new Blob([res.data]));

        // create "a" HTML element with href to file & click
        const link = document.createElement("a");
        link.href = url;
        link.setAttribute("download", "sample.wav"); //or any other extension
        document.body.appendChild(link);
        link.click();

        // clean up "a" element & remove ObjectURL
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        console.log(res);
      })
      .catch((error) => {
        console.log(error);
      });
  }
  return (
    <div>
      <div className="title">
        <h1>MariaBot</h1>
        <div className="title-caption">
          <h2>Malayalam TTS using Tacotron2 and Waveglow</h2>
        </div>
      </div>

      <div className="form-container">
        <form onSubmit={handleSubmit} className="text-form">
          <label>
            Type Here:{" "}
            <textarea
              name="myInput"
              type="text"
              value={inputText}
              placeholder="Type Manglish text here"
              onChange={(e) => setInputText(e.target.value)}
              className="input-field"
            />
          </label>
          <button type="submit" className="submit-button">Send</button>
        </form>
      </div>
    </div>
  );
}

export default App;

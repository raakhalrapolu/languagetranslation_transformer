import React, { useState } from 'react';
import './App.css';

function App() {
  const languages = ['Czech', 'English', 'Ukrainian', 'German'];
  const [inputText, setInputText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [inputLanguage, setInputLanguage] = useState('English'); // Use language codes
  const [outputLanguage, setOutputLanguage] = useState('German'); // Use language codes
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleLanguageChange = (event, type) => {
    const value = event.target.value;
    if (type === 'input') {
      setInputLanguage(value);
      if (value === outputLanguage) {
        setOutputLanguage(languages.find(lang => lang !== value));
      }
    } else {
      if (value !== inputLanguage) {
        setOutputLanguage(value);
      }
    }
  };

  const handleInputChange = (event) => {
    setInputText(event.target.value);
  };

  const handleTranslate = async () => {
    if (!inputText.trim()) {
      setError('Please enter some text to translate.');
      return;
    }

    setIsLoading(true);
    setError('');

    const translatePayload = {
      text: inputText,
      language_from: inputLanguage,
      language_to: outputLanguage
    };

    try {
      const response = await fetch("http://34.125.201.165:4200/translate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(translatePayload),
      });
      console.log(translatePayload);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setTranslatedText(data.translated_text); // Match the case with the API response
    } catch (error) {
      console.error('Error:', error);
      setError('Error fetching response');
    } finally {
      setIsLoading(false);
    }
  };

  const clearText = () => {
    setInputText('');
    setTranslatedText('');
    setError('');
  };

    return (
    <div className="App">
      <header>
        <h1>Language Translator</h1>
      </header>
      <div className="translator-container">
	<div className="language-inputs">
        <div className="input-section">
          <select value={inputLanguage} onChange={(e) => handleLanguageChange(e, 'input')}>
            {languages.map((lang, index) => (
                <option key={index} value={lang} disabled={lang === outputLanguage}>
                  {lang}
                </option>
            ))}
          </select>
          <textarea
            value={inputText}
            onChange={handleInputChange}
            placeholder="Enter text to translate"
          />
        </div>
        <div className="output-section">
          <select value={outputLanguage} onChange={(e) => handleLanguageChange(e, 'output')}>
            {languages.map((lang, index) => (
              <option key={index} value={lang} disabled={lang === inputLanguage}>
                {lang}
              </option>
            ))}
          </select>
          <textarea
            value={translatedText}
            className="translated-text"
            readOnly
          />
        </div>
       </div>
        <div className="button-container">
          <button onClick={handleTranslate} disabled={isLoading}>
            {isLoading ? 'Translating...' : 'Translate'}
          </button>
          <button onClick={clearText}>Clear</button>
        </div>
      </div>
    </div>
  );
}

export default App;

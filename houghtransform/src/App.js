// import logo from './logo.svg';
// import './App.css';
// import {useState,useEffect} from 'react';
// function App() {
//   const [state,setState] = useState({})

//   useEffect(()=>{
//     fetch("/api").then(response=>{
//       if (response.status==200){
//         return response.json()
//       }
//     }).then(data=>console.log(data))
//   });
//   return (
//     <div className="App">
//       <h1>hello</h1>
//     </div>
//   );
// }

// export default App;


import React, {useState} from 'react';
import './App.css';
import FileUploadPage from './components/FileUploadPage'

function App() {
  return (
    <div className="App">
      <header>
        <h1>Local Structure Similarity</h1>
      </header>
      {/* <Form setInputText={setInputText}/> */}
      <FileUploadPage></FileUploadPage>
    </div>
  );
}

export default App;

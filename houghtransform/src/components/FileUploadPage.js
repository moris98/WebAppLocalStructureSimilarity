import React, {useState} from 'react';

import { ReactSketchCanvas } from 'react-sketch-canvas';
// import InputRange from 'react-input-range';
// import Slider from '@mui/material/Slider';
import Image from 'react-bootstrap/Image';
import "bootstrap/dist/css/bootstrap.css";
import * as ReactBootStrap from 'react-bootstrap';

const FileUploadPage= () => {
	const [loading, setLoading] = useState(false);

	const [selectedFile='./black_screen.jpeg', setSelectedFile] = useState();
	const [isFilePicked, setIsFilePicked] = useState(false);

	const [heatmapResponseImage='./black_screen.jpeg', setHeatmapResponseImage] = useState(undefined)
	const [selectedSizeOfImage, setselectedSizeOfImage] = useState(64)
	
	const [isSketched, setIsSketched] = useState(false);
	const Sketch=React.createRef()

	const changeHandlerFileSelector = async (event) => {
		await setSelectedFile(event.target.files[0]);
		setIsFilePicked(true);
	}
	const changeHandlerSketch = (event) => {
		setIsSketched(true);
	}

	const clearSketch = async (event) =>{
		await Sketch.current.clearCanvas()
		setIsSketched(false);
	}

	const handleSelectedSizeChange = (event) => {
		setselectedSizeOfImage(event.target.value)
	}

	function urltoFile(url, filename, mimeType){
        return (fetch(url)
            .then(function(res){return res.arrayBuffer();})
            .then(function(buf){return new File([buf], filename,{type:mimeType});})
        );
    }
    //need to verify the concurent fetches - if first fall need to handle properly the second
	const handleSubmission = async () => {
		//checking if the needed images are loaded
		if (!isSketched || !isFilePicked){
			console.log("no sketch or no file input")
			return
		}
		//load the sketch
		let sketchJPG = await Sketch.current.exportImage("jpg")
		sketchJPG= await urltoFile(sketchJPG, 'sketchJPG.jpg','image/jpg')
		//create a FormData object with both the image to search in and the sketch
		const formData = new FormData();
		formData.append('fileToSearchIn', selectedFile);
		formData.append('sketch', sketchJPG);
		//set the loading state to true
		setLoading(true)
		//start fetching the FormData object
		fetch(
			// "https://mysterious-dusk-99830.herokuapp.com/uploadimage?picsize="+selectedSizeOfImage.toString(),
			"http://127.0.0.1:5000/uploadimage?picsize="+selectedSizeOfImage.toString(),
			{
				method: 'POST',
				body: formData
				
			}
			).then(async (response) => {//receive the first image - the objects that were found in the original image
				const blob = await response.blob()
				setSelectedFile(blob)
			}).then(()=>{//if first image was legal fetch the request for the second image - heat map
				fetch(
					// "https://mysterious-dusk-99830.herokuapp.com/heatmap",
					"http://127.0.0.1:5000/heatmap",
					{
						method: 'GET',
					}
					).then(async (response) => {//receive the second image - heat map
						const blob = await response.blob()
						setHeatmapResponseImage(URL.createObjectURL(blob))
						setLoading(false)
					}).catch(()=>{
						console.log("couldn't fetch the info.")
					});
					})
			.catch(()=>{
				console.log("couldn't fetch the info.")
			});
    }

	return (
        <div>
			<div className='form'>
				<input id='fileSelector' type="file" name="file" onChange={changeHandlerFileSelector} accept="image/png, image/jpeg"/>
				{/* <div id='sliderDiv'>20<Slider id='slider' aria-label="Temperatue" value={selectedSizeOfImage} onChange={handleSelectedSizeChange} valueLabelDisplay="on" min={20} max={150}/>150</div> */}
				<div id="patchSelectorDiv"> <label>Patch</label> <input type='number' onChange={handleSelectedSizeChange} min="10" max="200" id="patchSelector"/></div>
				<div className='sketchDiv' onClick={changeHandlerSketch}>
					<ReactSketchCanvas id="sketch" ref={Sketch} width="300px" height="300px" />
					<button onClick={clearSketch} disabled={loading}>Clear Sketch</button>
				</div>
				<div>
				{loading ?(<ReactBootStrap.Spinner animation="border" vid /> ):(<button onClick={handleSubmission}>Submit</button>)}
				</div>
			</div>
			<div id='presentedFile'>
			{isFilePicked  ? (
				<div id='presentResults'>
					<Image id='presentSelectedImage' src={URL.createObjectURL(selectedFile)}   thumbnail />
					<Image id='response' src={heatmapResponseImage} thumbnail />
				</div>
			) : (
				<div>
					<Image id='presentSelectedImage' src={selectedFile} thumbnail />
					{/* <Image id='response' src={locatedObjectsResponseImage} thumbnail /> */}
					<Image id='response' src={heatmapResponseImage} thumbnail />
				</div>
			)}
			</div>
        </div>
	);
}

export default FileUploadPage
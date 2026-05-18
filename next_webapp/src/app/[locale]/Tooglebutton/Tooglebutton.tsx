import { useState } from "react";


// This componen is for dark and light theme button
// It helps to change the theme of the page

const Tooglebutton = ({onValueChange}:any) => {
    const [darkvalue, setdarkvalue] = useState(false);
    const [dark_btn, setdark_btn] = useState("dark")

    const handleclick = () =>{
        if(darkvalue == false){
            setdarkvalue(true);
            setdark_btn("dark")
        }else{
            setdarkvalue(false);
            setdark_btn("light")
        }
        onValueChange(darkvalue);

    }

    return(
        <div className={`${darkvalue && "dark"}`}>
        <div className ="fixed bottom-4 right-4">
        <button  onClick={handleclick} className="bg-white dark:bg-white  text-black dark:text-black font-bold py-2 px-4 rounded-full shadow-lg  ">
            {dark_btn}
        </button>
        </div>
        </div>
    );
}

export default Tooglebutton;

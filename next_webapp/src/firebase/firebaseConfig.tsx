// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyAIOXy-29Qn4ruVUM_x3mYylOGLpPSDZrE",
  authDomain: "mop-next-webapp.firebaseapp.com",
  projectId: "mop-next-webapp",
  storageBucket: "mop-next-webapp.firebasestorage.app",
  messagingSenderId: "446436730978",
  appId: "1:446436730978:web:b63e054a54bb82faecc93b",
  measurementId: "G-4Q4DEL2YDC"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);
// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getFirestore, collection, addDoc } from 'firebase/firestore';
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyBq3czH7l3j10iMBTjdtVuI0zR2jIupA0o",
  authDomain: "openplayground-4a91d.firebaseapp.com",
  projectId: "openplayground-4a91d",
  storageBucket: "openplayground-4a91d.appspot.com",
  messagingSenderId: "387323207322",
  appId: "1:387323207322:web:301ef71134844f33060923",
  measurementId: "G-9C95XRXQ4P"
};



// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);
export { db, collection, addDoc };


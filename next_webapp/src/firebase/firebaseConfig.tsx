// This file holds the configuration on the firebase project
// Go to firebase console -> project settings
// Add the app you want to generate the credentials for

import { getApp, getApps, initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";
import { getStorage } from "firebase/storage";

// Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyDHqnuaPwnL7kY_Wr3CZaIJIit8pUMUQxg",
  authDomain: "mop-db-1e22a.firebaseapp.com",
  projectId: "mop-db-1e22a",
  storageBucket: "mop-db-1e22a.appspot.com",
  messagingSenderId: "436444330801",
  appId: "1:436444330801:web:2ce33edf0ac26fa60fce10"
};

// Initialize Firebase
const firebase = !getApps().length ? initializeApp(firebaseConfig) : getApp();
const auth = getAuth(firebase);
const db = getFirestore(firebase);
const storage = getStorage(firebase)

export { firebase, auth, db, storage }
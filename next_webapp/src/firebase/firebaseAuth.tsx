import { GoogleAuthProvider,signInWithPopup, onAuthStateChanged as _onAuthStateChanged, NextOrObserver, User } from "firebase/auth";
import { auth } from "./firebaseConfig";
import { useRouter } from "next/router";


// Auth state
export function onAuthStateChanged(cb: NextOrObserver<User>) {
    return _onAuthStateChanged(auth, cb);
}

// Sign in with google pop up and re-routes the user to protected app
export async function signInWithGoogle() {
    const provider = new GoogleAuthProvider();
    // const router = useRouter();

    try {
        const result = await signInWithPopup(auth, provider);
        const user = result.user as User;
        // router.push('/usecases'); // Change path to protected route
        console.log('User Signed in', user); 
    } 
    catch (error) {
        console.error("Error signing in with Google", error);
    }
}

// Handle signout
export async function signOutGoogle() {
    try {
        
        return auth.signOut();

    } catch (error) {
        console.error("Error signing out with Google", error);
    }
}

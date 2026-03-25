"use client";

import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import { useState, useEffect } from "react";
import { Camera, User, Mail, Phone, MapPin, Calendar, Save } from "lucide-react";

const Profile = () => {
  const [darkMode, setDarkMode] = useState(false);
  const [profileImage, setProfileImage] = useState(null);
  const [formData, setFormData] = useState({
    fullName: "",
    age: "",
    gender: "",
    email: "",
    phone: "",
    address: "",
    bio: "",
  });

  useEffect(() => {
    const root = document.documentElement;
    darkMode ? root.classList.add("dark") : root.classList.remove("dark");
  }, [darkMode]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setProfileImage(URL.createObjectURL(file));
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Profile Data:", formData);
    alert("Profile saved successfully!");
  };

  return (
    <div className="bg-gray-50 dark:bg-[#1d1919] text-black dark:text-white min-h-screen transition-colors duration-300">
      <Header />

      <section className="bg-gray-50 dark:bg-[#263238] py-12">
        <div className="max-w-6xl mx-auto px-4">
          <div className="mb-8 text-center">
            <h1 className="text-3xl md:text-4xl font-bold text-black dark:text-white">
              User Profile
            </h1>
            <p className="text-gray-600 dark:text-gray-300 mt-2">
              Create and manage your personal profile details.
            </p>
          </div>

          <div className="bg-white dark:bg-[#1f2a30] border border-gray-200 dark:border-white/10 rounded-2xl shadow-md p-6 md:p-8">
            <form onSubmit={handleSubmit} className="space-y-8">
              <div className="flex flex-col items-center justify-center">
                <div className="relative w-32 h-32 rounded-full overflow-hidden border-4 border-gray-200 dark:border-white/10 bg-white dark:bg-white/5 flex items-center justify-center">
                  {profileImage ? (
                    <img
                      src={profileImage}
                      alt="Profile Preview"
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <User className="w-14 h-14 text-gray-400" />
                  )}

                  <label className="absolute bottom-1 right-7 bg-green-600 hover:bg-green-700 text-white p-2 rounded-full cursor-pointer transition">
                    <Camera size={16} />
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleImageChange}
                      className="hidden"
                    />
                  </label>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-300 mt-3">
                  Upload profile picture
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                <div>
                  <label className="block mb-2 text-sm font-medium text-black dark:text-white">
                    Full Name
                  </label>
                  <div className="flex items-center bg-white dark:bg-white/5 border border-gray-300 dark:border-white/10 rounded-xl px-3">
                    <User size={18} className="text-gray-400 mr-2" />
                    <input
                      type="text"
                      name="fullName"
                      value={formData.fullName}
                      onChange={handleChange}
                      placeholder="Enter your full name"
                      className="w-full bg-transparent outline-none py-3 text-black dark:text-white placeholder-gray-500"
                    />
                  </div>
                </div>

                <div>
                  <label className="block mb-2 text-sm font-medium text-black dark:text-white">
                    Age
                  </label>
                  <div className="flex items-center bg-white dark:bg-white/5 border border-gray-300 dark:border-white/10 rounded-xl px-3">
                    <Calendar size={18} className="text-gray-400 mr-2" />
                    <input
                      type="number"
                      name="age"
                      value={formData.age}
                      onChange={handleChange}
                      placeholder="Enter your age"
                      className="w-full bg-transparent outline-none py-3 text-black dark:text-white placeholder-gray-500"
                    />
                  </div>
                </div>

                <div>
                  <label className="block mb-2 text-sm font-medium text-black dark:text-white">
                    Gender
                  </label>
                  <select
                    name="gender"
                    value={formData.gender}
                    onChange={handleChange}
                    className="w-full bg-white dark:bg-white/5 border border-gray-300 dark:border-white/10 rounded-xl px-4 py-3 outline-none text-black dark:text-white"
                  >
                    <option value="">Select gender</option>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                    <option value="Other">Other</option>
                  </select>
                </div>

                <div>
                  <label className="block mb-2 text-sm font-medium text-black dark:text-white">
                    Email
                  </label>
                  <div className="flex items-center bg-white dark:bg-white/5 border border-gray-300 dark:border-white/10 rounded-xl px-3">
                    <Mail size={18} className="text-gray-400 mr-2" />
                    <input
                      type="email"
                      name="email"
                      value={formData.email}
                      onChange={handleChange}
                      placeholder="Enter your email"
                      className="w-full bg-transparent outline-none py-3 text-black dark:text-white placeholder-gray-500"
                    />
                  </div>
                </div>

                <div>
                  <label className="block mb-2 text-sm font-medium text-black dark:text-white">
                    Phone
                  </label>
                  <div className="flex items-center bg-white dark:bg-white/5 border border-gray-300 dark:border-white/10 rounded-xl px-3">
                    <Phone size={18} className="text-gray-400 mr-2" />
                    <input
                      type="text"
                      name="phone"
                      value={formData.phone}
                      onChange={handleChange}
                      placeholder="Enter your phone number"
                      className="w-full bg-transparent outline-none py-3 text-black dark:text-white placeholder-gray-500"
                    />
                  </div>
                </div>

                <div>
                  <label className="block mb-2 text-sm font-medium text-black dark:text-white">
                    Address
                  </label>
                  <div className="flex items-center bg-white dark:bg-white/5 border border-gray-300 dark:border-white/10 rounded-xl px-3">
                    <MapPin size={18} className="text-gray-400 mr-2" />
                    <input
                      type="text"
                      name="address"
                      value={formData.address}
                      onChange={handleChange}
                      placeholder="Enter your address"
                      className="w-full bg-transparent outline-none py-3 text-black dark:text-white placeholder-gray-500"
                    />
                  </div>
                </div>
              </div>

              <div>
                <label className="block mb-2 text-sm font-medium text-black dark:text-white">
                  Bio
                </label>
                <textarea
                  name="bio"
                  value={formData.bio}
                  onChange={handleChange}
                  rows="5"
                  placeholder="Write something about yourself..."
                  className="w-full bg-white dark:bg-white/5 border border-gray-300 dark:border-white/10 rounded-xl px-4 py-3 outline-none text-black dark:text-white placeholder-gray-500"
                />
              </div>

              <div className="pt-2 text-center md:text-left">
                <button
                  type="submit"
                  className="inline-flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-xl font-semibold transition"
                >
                  <Save size={18} />
                  Save Profile
                </button>
              </div>
            </form>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Profile;
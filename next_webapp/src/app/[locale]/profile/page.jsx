"use client";

import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import { useState, useEffect } from "react";
import { Camera, User, Mail, Phone, MapPin, Calendar, Save } from "lucide-react";

const Profile = () => {
  const [darkMode, setDarkMode] = useState(false);
  const [profileImage, setProfileImage] = useState(null);
  const [formData, setFormData] = useState({
    first_name: "",
    last_name: "",
    age: "",
    gender: "",
    profile_img: "",
    email: "",
    phone: "",
    address: "",
    bio: "",
  });
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState("");

  useEffect(() => {
    const root = document.documentElement;
    darkMode ? root.classList.add("dark") : root.classList.remove("dark");
  }, [darkMode]);

  useEffect(() => {
    fetchProfile();
  }, []);

  const getUserId = () => {
    return localStorage.getItem("userId") || "1";
  };

  const fetchProfile = async () => {
    try {
      const userId = getUserId();
      const response = await fetch("/api/profile", {
        method: "GET",
        headers: { "x-user-id": userId },
      });

      const result = await response.json();
      if (result.success && result.data) {
        setFormData((prev) => ({
          ...prev,
          first_name: result.data.first_name || "",
          last_name: result.data.last_name || "",
          age: result.data.age || "",
          gender: result.data.gender || "",
          profile_img: result.data.profile_img || "",
          email: result.data.email || "",
          phone: result.data.phone || "",
          address: result.data.address || "",
          bio: result.data.bio || "",
        }));
      }
    } catch (error) {
      console.error("Failed to fetch profile:", error);
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
    if (errors[name]) {
      setErrors((prev) => ({ ...prev, [name]: "" }));
    }
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setProfileImage(URL.createObjectURL(file));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setErrors({});
    setSuccessMessage("");

    try {
      const userId = getUserId();
      const payload = {
        first_name: formData.first_name,
        last_name: formData.last_name,
        age: formData.age ? parseInt(formData.age) : undefined,
        gender: formData.gender,
        profile_img: formData.profile_img,
        email: formData.email,
        phone: formData.phone,
        address: formData.address,
        bio: formData.bio,
      };

      Object.keys(payload).forEach((key) =>
        payload[key] === undefined && delete payload[key]
      );

      const response = await fetch("/api/profile", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          "x-user-id": userId,
        },
        body: JSON.stringify(payload),
      });

      const result = await response.json();

      if (result.success) {
        setSuccessMessage("Profile updated successfully!");
        setTimeout(() => setSuccessMessage(""), 5000);
      } else {
        if (result.errors && Array.isArray(result.errors)) {
          const errorMap = {};
          result.errors.forEach((err) => {
            errorMap[err.field] = err.message;
          });
          setErrors(errorMap);
        } else {
          setErrors({ form: result.message });
        }
      }
    } catch (error) {
      console.error("Error updating profile:", error);
      setErrors({ form: "Failed to update profile. Please try again." });
    } finally {
      setLoading(false);
    }
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
            {successMessage && (
              <div className="bg-green-100 dark:bg-green-900 border border-green-400 dark:border-green-500 text-green-800 dark:text-green-100 px-4 py-3 rounded-lg mb-6">
                {successMessage}
              </div>
            )}

            {errors.form && (
              <div className="bg-red-100 dark:bg-red-900 border border-red-400 dark:border-red-500 text-red-800 dark:text-red-100 px-4 py-3 rounded-lg mb-6">
                {errors.form}
              </div>
            )}

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
                    First Name
                  </label>
                  <div
                    className={`flex items-center bg-white dark:bg-white/5 border rounded-xl px-3 ${
                      errors.first_name
                        ? "border-red-500"
                        : "border-gray-300 dark:border-white/10"
                    }`}
                  >
                    <User size={18} className="text-gray-400 mr-2" />
                    <input
                      type="text"
                      name="first_name"
                      value={formData.first_name}
                      onChange={handleChange}
                      placeholder="Enter your first name"
                      className="w-full bg-transparent outline-none py-3 text-black dark:text-white placeholder-gray-500"
                    />
                  </div>
                  {errors.first_name && (
                    <p className="text-red-500 text-sm mt-1">{errors.first_name}</p>
                  )}
                </div>

                <div>
                  <label className="block mb-2 text-sm font-medium text-black dark:text-white">
                    Last Name
                  </label>
                  <div
                    className={`flex items-center bg-white dark:bg-white/5 border rounded-xl px-3 ${
                      errors.last_name
                        ? "border-red-500"
                        : "border-gray-300 dark:border-white/10"
                    }`}
                  >
                    <User size={18} className="text-gray-400 mr-2" />
                    <input
                      type="text"
                      name="last_name"
                      value={formData.last_name}
                      onChange={handleChange}
                      placeholder="Enter your last name"
                      className="w-full bg-transparent outline-none py-3 text-black dark:text-white placeholder-gray-500"
                    />
                  </div>
                  {errors.last_name && (
                    <p className="text-red-500 text-sm mt-1">{errors.last_name}</p>
                  )}
                </div>

                <div>
                  <label className="block mb-2 text-sm font-medium text-black dark:text-white">
                    Age
                  </label>
                  <div
                    className={`flex items-center bg-white dark:bg-white/5 border rounded-xl px-3 ${
                      errors.age
                        ? "border-red-500"
                        : "border-gray-300 dark:border-white/10"
                    }`}
                  >
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
                  {errors.age && (
                    <p className="text-red-500 text-sm mt-1">{errors.age}</p>
                  )}
                </div>

                <div>
                  <label className="block mb-2 text-sm font-medium text-black dark:text-white">
                    Gender
                  </label>
                  <select
                    name="gender"
                    value={formData.gender}
                    onChange={handleChange}
                    className={`w-full bg-white dark:bg-white/5 border rounded-xl px-4 py-3 outline-none text-black dark:text-white ${
                      errors.gender
                        ? "border-red-500"
                        : "border-gray-300 dark:border-white/10"
                    }`}
                  >
                    <option value="">Select gender</option>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                    <option value="Other">Other</option>
                  </select>
                  {errors.gender && (
                    <p className="text-red-500 text-sm mt-1">{errors.gender}</p>
                  )}
                </div>

                <div>
                  <label className="block mb-2 text-sm font-medium text-black dark:text-white">
                    Email
                  </label>
                  <div
                    className={`flex items-center bg-white dark:bg-white/5 border rounded-xl px-3 ${
                      errors.email
                        ? "border-red-500"
                        : "border-gray-300 dark:border-white/10"
                    }`}
                  >
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
                  {errors.email && (
                    <p className="text-red-500 text-sm mt-1">{errors.email}</p>
                  )}
                </div>

                <div>
                  <label className="block mb-2 text-sm font-medium text-black dark:text-white">
                    Phone
                  </label>
                  <div
                    className={`flex items-center bg-white dark:bg-white/5 border rounded-xl px-3 ${
                      errors.phone
                        ? "border-red-500"
                        : "border-gray-300 dark:border-white/10"
                    }`}
                  >
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
                  {errors.phone && (
                    <p className="text-red-500 text-sm mt-1">{errors.phone}</p>
                  )}
                </div>

                <div>
                  <label className="block mb-2 text-sm font-medium text-black dark:text-white">
                    Address
                  </label>
                  <div
                    className={`flex items-center bg-white dark:bg-white/5 border rounded-xl px-3 ${
                      errors.address
                        ? "border-red-500"
                        : "border-gray-300 dark:border-white/10"
                    }`}
                  >
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
                  {errors.address && (
                    <p className="text-red-500 text-sm mt-1">{errors.address}</p>
                  )}
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
                  className={`w-full bg-white dark:bg-white/5 border rounded-xl px-4 py-3 outline-none text-black dark:text-white placeholder-gray-500 ${
                    errors.bio
                      ? "border-red-500"
                      : "border-gray-300 dark:border-white/10"
                  }`}
                />
                {errors.bio && (
                  <p className="text-red-500 text-sm mt-1">{errors.bio}</p>
                )}
              </div>

              <div className="pt-2 text-center md:text-left">
                <button
                  type="submit"
                  disabled={loading}
                  className="inline-flex items-center gap-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white px-6 py-3 rounded-xl font-semibold transition"
                >
                  <Save size={18} />
                  {loading ? "Saving..." : "Save Profile"}
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
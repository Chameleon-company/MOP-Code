"use client";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import "../../../../public/styles/contact.css";
import Image from "next/image";
import React, { useState, useEffect } from "react";
import { useTranslations } from "next-intl";
import { collection, addDoc } from "firebase/firestore";
import { db } from "../../../firebase/firebaseConfig";

interface FormField {
  name: string;
  spanName: string;
  type: string;
  placeholder: string;
  required: boolean;
  validator?: (value: string) => boolean;
}

const Contact = () => {
  const t = useTranslations("contact");

  const formFields: FormField[] = [
    {
      name: "firstName",
      spanName: t("First Name"),
      type: "text",
      placeholder: t("Enter Your First name"),
      required: true,
      validator: (value: string) => value.trim() !== "",
    },
    {
      name: "lastName",
      spanName: t("Last Name"),
      type: "text",
      placeholder: t("Enter Your Last name"),
      required: true,
      validator: (value: string) => value.trim() !== "",
    },
    {
      name: "email",
      spanName: t("Company Email Address"),
      type: "email",
      placeholder: t("Enter Company Email Address"),
      required: true,
      validator: (email: string) => /^\S+@\S+\.\S+$/.test(email),
    },
    {
      name: "phone",
      spanName: t("Phone Number"),
      type: "tel",
      placeholder: t("Enter Your Phone Number"),
      required: true,
      validator: (phone: string) => /^\d{10,}$/.test(phone.replace(/\D/g, "")),
    },
    {
      name: "message",
      spanName: t("What can I help you with"),
      type: "textarea",
      placeholder: t("Enter Message"),
      required: true,
      validator: (value: string) => value.trim() !== "",
    },
  ];

  const [formValues, setFormValues] = useState<{ [key: string]: string }>({});
  const [errors, setErrors] = useState<{ [key: string]: string }>({});
  const [successMessage, setSuccessMessage] = useState("");
  const [failureMessage, setFailureMessage] = useState("");
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    const theme = localStorage.getItem("theme");
    if (theme === "dark") {
      setDarkMode(true);
      document.documentElement.classList.add("dark");
    } else {
      setDarkMode(false);
      document.documentElement.classList.remove("dark");
    }
  }, []);

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    const { name, value } = e.target;
    setFormValues({ ...formValues, [name]: value });
    if (errors[name]) validateField(name, value);
  };

  const validateField = (name: string, value: string) => {
    const field = formFields.find((f) => f.name === name);
    if (field?.validator && !field.validator(value)) {
      setErrors({
        ...errors,
        [name]: `Invalid ${field.spanName.toLowerCase()}`,
      });
    } else {
      const newErrors = { ...errors };
      delete newErrors[name];
      setErrors(newErrors);
    }
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    let valid = true;
    const newErrors: { [key: string]: string } = {};

    formFields.forEach((field) => {
      const value = formValues[field.name] || "";
      if (field.validator && !field.validator(value)) {
        newErrors[field.name] = `Invalid ${field.spanName.toLowerCase()}`;
        valid = false;
      }
    });

    setErrors(newErrors);
    if (!valid) return;

    try {
      await addDoc(collection(db, "contacts"), formValues);
      setSuccessMessage("Form submitted successfully!");
      setFailureMessage("");
      setFormValues({});
    } catch (error) {
      setFailureMessage("Failed to submit form. Please try again.");
      setSuccessMessage("");
    }
  };

  return (
    <div className="bg-white dark:bg-black min-h-screen font-sans text-black dark:text-white transition-colors duration-300">
      <Header />

      <main className="max-w-7xl mx-auto px-6 py-16 flex flex-col lg:flex-row gap-10">
        <div className="w-full lg:w-1/2">
          {successMessage && (
            <p className="text-green-600 dark:text-green-400 text-sm mb-3">
              {successMessage}
            </p>
          )}
          {failureMessage && (
            <p className="text-red-600 dark:text-red-400 text-sm mb-3">
              {failureMessage}
            </p>
          )}
          <form onSubmit={handleSubmit} noValidate className="space-y-5">
            {formFields.map((field) => (
              <div key={field.name}>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  {field.spanName}
                </label>
                {field.type === "textarea" ? (
                  <textarea
                    name={field.name}
                    placeholder={field.placeholder}
                    required={field.required}
                    className="w-full border border-black dark:border-white bg-white dark:bg-gray-900 text-black dark:text-white rounded-md p-3 text-sm focus:ring-2 focus:ring-green-400 focus:outline-none h-24"
                    onChange={handleChange}
                  />
                ) : (
                  <input
                    name={field.name}
                    type={field.type}
                    placeholder={field.placeholder}
                    required={field.required}
                    className="w-full border border-black dark:border-white bg-white dark:bg-gray-900 text-black dark:text-white rounded-md p-3 text-sm focus:ring-2 focus:ring-green-400 focus:outline-none"
                    onChange={handleChange}
                  />
                )}
                {errors[field.name] && (
                  <p className="text-red-500 dark:text-red-400 text-xs mt-1">
                    {errors[field.name]}
                  </p>
                )}
              </div>
            ))}
            <div className="flex justify-center pt-2">
              <button
                type="submit"
                className="bg-green-600 hover:bg-green-900 text-white uppercase font-bold text-sm px-8 py-3 rounded-md transition duration-200"
              >
                {t("Submit")}
              </button>
            </div>
          </form>
        </div>

        <div className="w-full lg:w-1/2 bg-[#F0F0F0] dark:bg-gray-800 p-8 rounded-lg shadow-md dark:shadow-none">
          <h2 className="text-4xl font-bold text-black dark:text-white mb-4 text-center">
            Contact Us
          </h2>
          <p className="text-sm text-gray-600 dark:text-gray-300 mb-6">
            Feel free to use the form or drop us an email
          </p>
          <div className="mb-4">
            <div className="flex items-center text-sm text-gray-800 dark:text-gray-200 mb-2">
              <span className="mr-3">📧</span>
              <span>email@example.com</span>
            </div>
            <hr className="mb-4 border-gray-300 dark:border-gray-600" />
            <div className="flex items-center text-sm text-gray-800 dark:text-gray-200">
              <span className="mr-3">📞</span>
              <span>+61 123 456 789</span>
            </div>
          </div>
          <iframe
            src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d805190.2361230071!2d144.3937342027546!3d-37.97072605426427!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x6ad646b5d2ba4df7%3A0x4045675218ccd90!2sMelbourne%20VIC!5e0!3m2!1sen!2sau!4v1747916966526!5m2!1sen!2sau"
            height="400"
            className="mapIframe"
            allowFullScreen={true}
            style={{ border: 0 }}
            loading="lazy"
            referrerPolicy="no-referrer-when-downgrade"
            title="Google Maps Embed"
          ></iframe>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default Contact;

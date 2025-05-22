"use client";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import "../../../../public/styles/contact.css";
import Image from "next/image";
import React, { useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import { collection, addDoc } from "firebase/firestore";
import { db } from "../../../firebase/firebaseConfig";
import { Mail, Phone } from "lucide-react";

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
      setErrors({ ...errors, [name]: `Invalid ${field.spanName.toLowerCase()}` });
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
  const[darkMode,setDarkMode]=useState(false)
    useEffect(() => {
        const htmlElement = document.documentElement;
        const hasDarkClass = htmlElement.classList.contains("dark");
        setDarkMode(hasDarkClass);
    }, []);

  return (<div className="contactPage font-sans bg-white min-h-screen dark:bg-black">
    <Header />
    <main className="contactBody font-light text-xs leading-7 flex flex-col lg:flex-row lg:space-x-8 mt-12 items-start p-12 dark:bg-black">


    <div className="imgContent relative w-full lg:w-1/2 mt-12 order-1 lg:order-2 p-6 bg-[#d9d9d9] dark:bg-[#666666]">
    <span className="block text-4xl font-bold leading-snug font-montserrat text-black dark:text-white text-center -mt-6 pl-6 lg:pl-0 lg:mt-10 lg:mb-8 lg:text-center z-10 relative lg:top-0 lg:left-0">
    {t("Contact")} { }
         
          {t("Us")}
        </span>
        <div style={{  marginBottom: '16px' }}>
          <span className="dark:text-white font-bold text-[15px]" style={{ marginLeft: '85px' }}>{"Feel free to use the form or drop us an email"}</span>
        </div>
        <div className="flex flex-col gap-4 w-full max-w-md">
      {/* Email Field */}
      <div className="flex items-center border-b border-gray-300 px-3 py-2 w-full dark:text-white" style={{ marginLeft: '70px' }}>
  <Mail className="text-gray-500 mr-2" />
  <input
    type="email"
    placeholder="Email"
    className="outline-none w-full bg-transparent"
  />
</div>

      {/* Phone Field */}
      <div className="flex items-center border-b border-gray-300 px-3 py-2 w-full dark:text-white" style={{ marginBottom: '40px',marginLeft: '70px' }}>
        <Phone className="text-gray-500 mr-2" />
        <input
          type="tel"
          placeholder="Phone Number"
          className="outline-none w-full bg-transparent"
        />
      </div>
    </div>
    <div className="imgWrap relative w-full mt-4 lg:mt-0 flex justify-center items-center mt-[50px]">
    <Image
            src="/img/map.png"
            alt="City"
            width={600}
            height={400}
            className="cityImage block w-200 h-auto "
          />
        </div>


        
      </div>
      

      <div className="formContent w-full lg:w-1/2 order-2 lg:order-1 lg:pr-8 ">
        {successMessage && <p className="text-green-500 mb-3">{successMessage}</p>}
        {failureMessage && <p className="text-red-500 mb-3">{failureMessage}</p>}
        <form
          id="contact"
          action=""
          onSubmit={handleSubmit}
          method="post"
          className="m-8"
          noValidate
        >
          {formFields.map((field) => (
            <fieldset
              key={field.name}
              className="border-0 m-0 mb-2.5 min-w-full p-0 w-full text-gray-700"
            >
              <span className="namaSpan text-black dark:text-white">{field.spanName}</span>
              {field.type === "textarea" ? (
                <textarea
                  name={field.name}
                  placeholder={field.placeholder}
                  required={field.required}
                  className="w-full border border-gray-300 bg-white mb-1 p-2.5 font-normal text-xs rounded-md focus:border-gray-400 transition-colors ease-in-out duration-300 h-16"
                  onChange={handleChange}
                ></textarea>
              ) : (
                <input
                  name={field.name}
                  type={field.type}
                  placeholder={field.placeholder}
                  required={field.required}
                  className="w-full border border-gray-300 bg-white mb-1 p-2.5 font-normal text-xs rounded-md focus:border-gray-400 transition-colors ease-in-out duration-300"
                  onChange={handleChange}
                />
              )}
              {errors[field.name] && (
                <span className="text-red-500 text-xs">
                  {errors[field.name]}
                </span>
              )}
            </fieldset>
          ))}
          <div className="flex justify-center items-center">
            <button className="bg-green-800 text-white font-semibold text-lg py-1 px-6 rounded hover:bg-green-800 focus:outline-none focus:ring-2 focus:ring-green-700 focus:ring-opacity-50 ">
              {t("Submit")}
            </button>

  return (
    <div className="bg-white min-h-screen font-sans">
      <Header />
      
      <main className="max-w-7xl mx-auto px-6 py-16 flex flex-col lg:flex-row gap-10">
        <div className="w-full lg:w-1/2">
          {successMessage && (
            <p className="text-green-600 text-sm mb-3">{successMessage}</p>
          )}
          {failureMessage && (
            <p className="text-red-600 text-sm mb-3">{failureMessage}</p>
          )}
          <form onSubmit={handleSubmit} noValidate className="space-y-5">
            {formFields.map((field) => (
              <div key={field.name}>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {field.spanName}
                </label>
                {field.type === "textarea" ? (
                  <textarea
                    name={field.name}
                    placeholder={field.placeholder}
                    required={field.required}
                    className="w-full border border-black rounded-md p-3 text-sm focus:ring-2 focus:ring-green-400 focus:outline-none h-24" // 🔧 Changed border to black
                    onChange={handleChange}
                  />
                ) : (
                  <input
                    name={field.name}
                    type={field.type}
                    placeholder={field.placeholder}
                    required={field.required}
                    className="w-full border border-black rounded-md p-3 text-sm focus:ring-2 focus:ring-green-400 focus:outline-none" // 🔧 Changed border to black
                    onChange={handleChange}
                  />
                )}
                {errors[field.name] && (
                  <p className="text-red-500 text-xs mt-1">
                    {errors[field.name]}
                  </p>
                )}
              </div>
            ))}
            <div className="flex justify-center pt-2">
              <button
                type="submit"
                className="bg-green-600 hover:bg-green-900 text-white uppercase font-bold text-sm px-8 py-3 rounded-md transition duration-200" // 🔧 Submit button: bold, uppercase, centered
              >
                {t("Submit")}
              </button>
            </div>
          </form>
        </div>

        <div className="w-full lg:w-1/2 bg-[#F0F0F0] p-8 rounded-lg shadow-md">
          <h2 className="text-4xl font-bold text-black mb-4 text-center">Contact Us</h2>
          <p className="text-sm text-gray-600 mb-6">
            Feel free to use the form or drop us an email
          </p>
          <div className="mb-4">
            <div className="flex items-center text-sm text-gray-800 mb-2">
              <span className="mr-3">📧</span>
              <span>email@example.com</span>
            </div>
            <hr className="mb-4 border-gray-300" /> 
            <div className="flex items-center text-sm text-gray-800">
              <span className="mr-3">📞</span>
              <span>+61 123 456 789</span>
            </div>
          </div>
          <iframe
            src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3153.019970066525!2d144.96145431531744!3d-37.81410797975166!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x6ad65d43f37c3e3f%3A0x5045675218ce6e0!2sMelbourne!5e0!3m2!1sen!2sau!4v1588166683930!5m2!1sen!2sau"
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

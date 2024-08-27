"use client";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import "../../../../public/styles/contact.css";
import Image from "next/image";
import React, { useState } from "react";
import { useTranslations } from "next-intl";
import Tooglebutton from "../Tooglebutton/Tooglebutton";
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

  const formFields = [
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
      spanName: t("Message"),
      type: "textarea",
      placeholder: t("Enter Message"),
      required: true,
      validator: (value: string) => value.trim() !== "",
    },
  ];
  const [formValues, setFormValues] = useState<{ [key: string]: string }>({});
  const [errors, setErrors] = useState<{ [key: string]: string }>({});

  // Handler to update form values and clear errors
  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    const { name, value } = e.target;
    setFormValues({
      ...formValues,
      [name]: value,
    });
    // If there's an error previously, validate again and clear if resolved
    if (errors[name]) {
      validateField(name, value);
    }
  };

  // Function to validate individual fields
  const validateField = (name: string, value: string) => {
    const field = formFields.find((field) => field.name === name);
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

  // Form submission handler
  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    let valid = true;
    // Validate all fields before submitting
    formFields.forEach((field) => {
      if (field.validator && !field.validator(formValues[field.name] || "")) {
        setErrors((prev) => ({
          ...prev,
          [field.name]: `Invalid ${field.spanName.toLowerCase()}`,
        }));
        valid = false;
      }
    });
  };

  //dark theme
  const [dark_value,setdarkvalue] = useState(false);
  
  const handleValueChange = (newValue: boolean | ((prevState: boolean) => boolean))=>{
    setdarkvalue(newValue);
  }

  return (
    <div className= {`${dark_value && "dark"}`}>
    <div className="contactPage font-sans bg-gray-200 min-h-screen dark:bg-pr_bg_dark">
      <Header />
      <Tooglebutton onValueChange={handleValueChange}/>
      <main className="contactBody font-light text-xs leading-7 flex flex-col justify-between mt-12 items-start p-12 dark:bg-pr_bg_dark">
        <div className="formContent w-full">
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
                className="border-0 m-0 mb-2.5 min-w-full p-0 w-full text-gray-700 dark:bg-pr_bg_dark "
              >
                <span className="namaSpan text-black dark:text-white">{field.spanName}</span>
                {field.type === "textarea" ? (
                  <textarea
                    name={field.name}
                    placeholder={field.placeholder}
                    required={field.required}
                    className="w-full border border-gray-300 bg-white mb-1 p-2.5 font-normal text-xs rounded-md focus:border-gray-400 transition-colors ease-in-out duration-300 h-16 dark:bg-sc_bg_dark"
                    onChange={handleChange}
                  ></textarea>
                ) : (
                  <input
                    name={field.name}
                    type={field.type}
                    placeholder={field.placeholder}
                    required={field.required}
                    className="w-full border border-gray-300 bg-white mb-1 p-2.5 font-normal text-xs rounded-md focus:border-gray-400 transition-colors ease-in-out duration-300 dark:bg-sc_bg_dark"
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
            </div>
          </form>
        </div>

        <div className="imgContent max-w-full max-h-full text-center relative w-full mt-12">
          <span className="contactUsText absolute text-left  text-black text-3xl leading-snug font-montserrat dark:text-white">
            {t("Contact")}
            <br />
            {t("Us")}
          </span>
          <div className="imgWrap relative inline-block">
            <Image
              src="/img/contact-us-city.png"
              alt="City"
              width={800}
              height={600}
              className="cityImage block relative z-10"
            />
          </div>
        </div>
      </main>
      <Footer />
    </div>
    </div>
  );
};

export default Contact;

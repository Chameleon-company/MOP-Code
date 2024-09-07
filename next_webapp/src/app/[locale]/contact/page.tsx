"use client";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import "../../../../public/styles/contact.css";
import Image from "next/image";
import React, { useState } from "react";
import { useTranslations } from "next-intl";


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
  const [successMessage, setSuccessMessage] = useState("");
  const [failureMessage, setFailureMessage] = useState("");

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
  const handleSubmit =async (e: React.FormEvent<HTMLFormElement>) => {
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

  return (<div className="contactPage font-sans bg-white min-h-screen">
    <Header />
    <main className="contactBody font-light text-xs leading-7 flex flex-col lg:flex-row lg:space-x-8 mt-12 items-start p-12">
  
      <div className="imgContent relative w-full lg:w-1/2 mt-12 order-1 lg:order-2">
      <span className="contactUsText block text-black text-4xl font-normal leading-snug font-montserrat mt-6 pl-6 text-left lg:absolute lg:left-0 lg:top-0 lg:transform lg:translate-y-[-10%] lg:pl-0 lg:text-center lg:mb-8 z-20">
  {t("Contact")}
  <br />
  {t("Us")}
</span>

        <div className="imgWrap relative w-full mt-4 lg:mt-0">
          <Image
            src="/img/contact-us-city.png"
            alt="City"
            width={800}
            height={600}
            className="cityImage block w-full h-auto"
          />
        </div>
      </div>
  
      <div className="formContent w-full lg:w-1/2 order-2 lg:order-1 lg:pr-8">
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
              <span className="namaSpan text-black">{field.spanName}</span>
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
          </div>
        </form>
      </div>
  
    </main>
    <Footer />
  </div>
  

  );
};

export default Contact;
"use client";

import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import React, { useEffect, useMemo, useState } from "react";
import { useTranslations } from "next-intl";
import { collection, addDoc, serverTimestamp } from "firebase/firestore";
import { db } from "../../../firebase/firebaseConfig";
import { FiMail, FiMapPin, FiPhone, FiSend } from "react-icons/fi";

type FormField = {
  name: string;
  label: string;
  type: "text" | "email" | "tel" | "textarea";
  placeholder: string;
  required: boolean;
  validator?: (value: string) => boolean;
  errorMessage: string;
};

type FormValues = Record<string, string>;
type FormErrors = Record<string, string>;

const Contact = () => {
  const t = useTranslations("contact");

  const formFields: FormField[] = useMemo(
    () => [
      {
        name: "firstName",
        label: t("First Name"),
        type: "text",
        placeholder: t("Enter Your First name"),
        required: true,
        validator: (value) => value.trim().length >= 2,
        errorMessage: "Please enter a valid first name.",
      },
      {
        name: "lastName",
        label: t("Last Name"),
        type: "text",
        placeholder: t("Enter Your Last name"),
        required: true,
        validator: (value) => value.trim().length >= 2,
        errorMessage: "Please enter a valid last name.",
      },
      {
        name: "email",
        label: t("Email Address"),
        type: "email",
        placeholder: t("Enter Your Email Address"),
        required: true,
        validator: (email) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email),
        errorMessage: "Please enter a valid email address.",
      },
      {
        name: "phone",
        label: t("Phone Number"),
        type: "tel",
        placeholder: t("Enter Your Phone Number"),
        required: true,
        validator: (phone) => /^\+?[\d\s()-]{8,}$/.test(phone.trim()),
        errorMessage: "Please enter a valid phone number.",
      },
      {
        name: "subject",
        label: "Subject",
        type: "text",
        placeholder: "Enter subject",
        required: true,
        validator: (value) => value.trim().length >= 3,
        errorMessage: "Please enter a subject.",
      },
      {
        name: "message",
        label: t("What can I help you with"),
        type: "textarea",
        placeholder: t("Enter Message"),
        required: true,
        validator: (value) => value.trim().length >= 10,
        errorMessage: "Message must be at least 10 characters long.",
      },
    ],
    [t]
  );

  const initialValues = useMemo(
    () =>
      formFields.reduce<FormValues>((acc, field) => {
        acc[field.name] = "";
        return acc;
      }, {}),
    [formFields]
  );

  const [formValues, setFormValues] = useState<FormValues>(initialValues);
  const [errors, setErrors] = useState<FormErrors>({});
  const [successMessage, setSuccessMessage] = useState("");
  const [failureMessage, setFailureMessage] = useState("");
  const [darkMode, setDarkMode] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);

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

  useEffect(() => {
    setFormValues(initialValues);
  }, [initialValues]);

  useEffect(() => {
    if (!showSuccess) return;

    const timer = setTimeout(() => {
      setShowSuccess(false);
      setSuccessMessage("");
    }, 3000);

    return () => clearTimeout(timer);
  }, [showSuccess]);

  const validateField = (name: string, value: string) => {
    const field = formFields.find((item) => item.name === name);
    if (!field) return;

    if (field.required && field.validator && !field.validator(value)) {
      setErrors((prev) => ({
        ...prev,
        [name]: field.errorMessage,
      }));
      return;
    }

    setErrors((prev) => {
      const next = { ...prev };
      delete next[name];
      return next;
    });
  };

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    const { name, value } = e.target;

    setFormValues((prev) => ({
      ...prev,
      [name]: value,
    }));

    if (errors[name]) {
      validateField(name, value);
    }
  };

  const handleBlur = (
    e: React.FocusEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    const { name, value } = e.target;
    validateField(name, value);
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
  e.preventDefault();

  const newErrors: FormErrors = {};

  formFields.forEach((field) => {
    const value = formValues[field.name] || "";
    if (field.required && field.validator && !field.validator(value)) {
      newErrors[field.name] = field.errorMessage;
    }
  });

  setErrors(newErrors);
  setFailureMessage("");

  if (Object.keys(newErrors).length > 0) {
    return;
  }

  try {
    setIsSubmitting(true);

    await new Promise((resolve) => setTimeout(resolve, 1000));

    setSuccessMessage("Your message has been sent successfully.");
    setShowSuccess(true);
    setFormValues(initialValues);
    setErrors({});
  } catch (error) {
    setFailureMessage("Failed to submit the form. Please try again.");
  } finally {
    setIsSubmitting(false);
  }
};

  return (
    <div className="min-h-screen bg-gradient-to-b from-white via-green-50/50 to-white font-sans text-black transition-colors duration-300 dark:from-black dark:via-gray-950 dark:to-black dark:text-white">
      <Header />

      <main className="mx-auto max-w-7xl px-6 py-14 lg:px-8">
        <div className="mb-10 text-center">
          <h1 className="mt-4 text-4xl font-bold tracking-tight text-gray-900 dark:text-white sm:text-5xl">
            Let’s Start a Conversation
          </h1>
          <p className="mx-auto mt-4 max-w-2xl text-base text-gray-600 dark:text-gray-300 sm:text-lg">
            Have a question, suggestion, or collaboration idea? Fill out the
            form and our team will get back to you as soon as possible.
          </p>
        </div>

        <div className="grid gap-8 lg:grid-cols-[1.15fr_0.85fr]">
          <section className="relative overflow-hidden rounded-3xl border border-gray-200 bg-white p-6 shadow-xl dark:border-gray-800 dark:bg-gray-900 sm:p-8">
            <div className="mb-6 flex items-center justify-between gap-4">
              <div>
                <h2 className="text-2xl font-semibold text-gray-900 dark:text-white">
                  Send us a message
                </h2>
                <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                  Please complete all required fields.
                </p>
              </div>
            </div>

            <div
              className={`mb-5 transform rounded-2xl border border-green-200 bg-green-50 px-4 py-3 text-sm font-medium text-green-700 shadow-sm transition-all duration-500 dark:border-green-800 dark:bg-green-950/40 dark:text-green-300 ${
                showSuccess
                  ? "translate-y-0 opacity-100"
                  : "pointer-events-none -translate-y-2 opacity-0"
              }`}
            >
              {successMessage}
            </div>

            {failureMessage && (
              <div className="mb-5 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-medium text-red-600 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300">
                {failureMessage}
              </div>
            )}

            <form onSubmit={handleSubmit} noValidate className="space-y-6">
              <div className="grid gap-6 sm:grid-cols-2">
                {formFields.slice(0, 2).map((field) => (
                  <div key={field.name} className="space-y-2">
                    <label
                      htmlFor={field.name}
                      className="block text-sm font-medium text-gray-700 dark:text-gray-300"
                    >
                      {field.label}
                    </label>
                    <input
                      id={field.name}
                      name={field.name}
                      type={field.type}
                      placeholder={field.placeholder}
                      value={formValues[field.name] || ""}
                      onChange={handleChange}
                      onBlur={handleBlur}
                      className={`w-full rounded-2xl border bg-white px-4 py-3 text-sm text-gray-900 outline-none transition placeholder:text-gray-400 focus:ring-2 dark:bg-gray-950 dark:text-white ${
                        errors[field.name]
                          ? "border-red-400 focus:ring-red-200 dark:border-red-500"
                          : "border-gray-300 focus:border-green-500 focus:ring-green-200 dark:border-gray-700"
                      }`}
                    />
                    {errors[field.name] && (
                      <p className="text-xs text-red-500">{errors[field.name]}</p>
                    )}
                  </div>
                ))}
              </div>

              <div className="grid gap-6 sm:grid-cols-2">
                {formFields.slice(2, 4).map((field) => (
                  <div key={field.name} className="space-y-2">
                    <label
                      htmlFor={field.name}
                      className="block text-sm font-medium text-gray-700 dark:text-gray-300"
                    >
                      {field.label}
                    </label>
                    <input
                      id={field.name}
                      name={field.name}
                      type={field.type}
                      placeholder={field.placeholder}
                      value={formValues[field.name] || ""}
                      onChange={handleChange}
                      onBlur={handleBlur}
                      className={`w-full rounded-2xl border bg-white px-4 py-3 text-sm text-gray-900 outline-none transition placeholder:text-gray-400 focus:ring-2 dark:bg-gray-950 dark:text-white ${
                        errors[field.name]
                          ? "border-red-400 focus:ring-red-200 dark:border-red-500"
                          : "border-gray-300 focus:border-green-500 focus:ring-green-200 dark:border-gray-700"
                      }`}
                    />
                    {errors[field.name] && (
                      <p className="text-xs text-red-500">{errors[field.name]}</p>
                    )}
                  </div>
                ))}
              </div>

              <div className="space-y-2">
                <label
                  htmlFor="subject"
                  className="block text-sm font-medium text-gray-700 dark:text-gray-300"
                >
                  Subject
                </label>
                <input
                  id="subject"
                  name="subject"
                  type="text"
                  placeholder="Enter subject"
                  value={formValues.subject || ""}
                  onChange={handleChange}
                  onBlur={handleBlur}
                  className={`w-full rounded-2xl border bg-white px-4 py-3 text-sm text-gray-900 outline-none transition placeholder:text-gray-400 focus:ring-2 dark:bg-gray-950 dark:text-white ${
                    errors.subject
                      ? "border-red-400 focus:ring-red-200 dark:border-red-500"
                      : "border-gray-300 focus:border-green-500 focus:ring-green-200 dark:border-gray-700"
                  }`}
                />
                {errors.subject && (
                  <p className="text-xs text-red-500">{errors.subject}</p>
                )}
              </div>

              <div className="space-y-2">
                <label
                  htmlFor="message"
                  className="block text-sm font-medium text-gray-700 dark:text-gray-300"
                >
                  {t("What can I help you with")}
                </label>
                <textarea
                  id="message"
                  name="message"
                  rows={6}
                  placeholder={t("Enter Message")}
                  value={formValues.message || ""}
                  onChange={handleChange}
                  onBlur={handleBlur}
                  className={`w-full rounded-2xl border bg-white px-4 py-3 text-sm text-gray-900 outline-none transition placeholder:text-gray-400 focus:ring-2 dark:bg-gray-950 dark:text-white ${
                    errors.message
                      ? "border-red-400 focus:ring-red-200 dark:border-red-500"
                      : "border-gray-300 focus:border-green-500 focus:ring-green-200 dark:border-gray-700"
                  }`}
                />
                {errors.message && (
                  <p className="text-xs text-red-500">{errors.message}</p>
                )}
              </div>

              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  By submitting this form, you agree to be contacted regarding
                  your enquiry.
                </p>

                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="inline-flex items-center justify-center gap-2 rounded-2xl bg-green-600 px-6 py-3 text-sm font-semibold text-white shadow-lg transition duration-300 hover:bg-green-700 disabled:cursor-not-allowed disabled:opacity-70"
                >
                  <FiSend className="text-base" />
                  {isSubmitting ? "Sending..." : t("Submit")}
                </button>
              </div>
            </form>
          </section>

          <aside className="rounded-3xl bg-gradient-to-br from-green-600 to-green-800 p-8 text-white shadow-xl">
            <h2 className="text-3xl font-bold">Contact Information</h2>
            <p className="mt-3 text-sm leading-6 text-green-50">
              Prefer to reach us directly? Use the details below or send your
              enquiry through the form.
            </p>

            <div className="mt-8 space-y-4">
              <div className="flex items-start gap-4 rounded-2xl bg-white/10 p-4 backdrop-blur-sm">
                <FiMail className="mt-1 text-xl" />
                <div>
                  <p className="text-sm font-medium text-green-100">Email</p>
                  <p className="text-base font-semibold">email@example.com</p>
                </div>
              </div>

              <div className="flex items-start gap-4 rounded-2xl bg-white/10 p-4 backdrop-blur-sm">
                <FiPhone className="mt-1 text-xl" />
                <div>
                  <p className="text-sm font-medium text-green-100">Phone</p>
                  <p className="text-base font-semibold">+61 123 456 789</p>
                </div>
              </div>

              <div className="flex items-start gap-4 rounded-2xl bg-white/10 p-4 backdrop-blur-sm">
                <FiMapPin className="mt-1 text-xl" />
                <div>
                  <p className="text-sm font-medium text-green-100">Location</p>
                  <p className="text-base font-semibold">Melbourne, Australia</p>
                </div>
              </div>
            </div>

            <div className="mt-8 overflow-hidden rounded-2xl border border-white/15 shadow-lg">
              <iframe
                src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d805190.2361230071!2d144.3937342027546!3d-37.97072605426427!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x6ad646b5d2ba4df7%3A0x4045675218ccd90!2sMelbourne%20VIC!5e0!3m2!1sen!2sau!4v1747916966526!5m2!1sen!2sau"
                height="320"
                className="w-full"
                allowFullScreen
                style={{ border: 0 }}
                loading="lazy"
                referrerPolicy="no-referrer-when-downgrade"
                title="Google Maps Embed"
              />
            </div>
          </aside>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default Contact;
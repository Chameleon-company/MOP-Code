"use client";

import { useState } from "react";
import { Mail, MapPin, Phone } from "lucide-react";

type FormDataType = {
  fullName: string;
  email: string;
  subject: string;
  message: string;
};

type FormErrorsType = {
  fullName?: string;
  email?: string;
  subject?: string;
  message?: string;
};

export default function ContactUsSection() {
  const [formData, setFormData] = useState<FormDataType>({
    fullName: "",
    email: "",
    subject: "",
    message: "",
  });

  const [errors, setErrors] = useState<FormErrorsType>({});
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [successMessage, setSuccessMessage] = useState("");
  const [failureMessage, setFailureMessage] = useState("");

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    const { name, value } = e.target;
  
    const updatedValue = name === "message" ? value.slice(0, 255) : value;
  
    setFormData((prev) => ({
      ...prev,
      [name]: updatedValue,
    }));
  
    
    if (name === "email") {
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  
      if (!updatedValue.trim()) {
        setErrors((prev) => ({
          ...prev,
          email: "Email is required",
        }));
      } else if (!emailRegex.test(updatedValue)) {
        setErrors((prev) => ({
          ...prev,
          email: "Please enter a valid email address",
        }));
      } else {
        setErrors((prev) => {
          const newErrors = { ...prev };
          delete newErrors.email;
          return newErrors;
        });
      }
  
      return;
    }
  
    
    setErrors((prev) => ({
      ...prev,
      [name]: "",
    }));
  };

  const validateForm = () => {
    const newErrors: FormErrorsType = {};
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

    if (!formData.fullName.trim()) {
      newErrors.fullName = "Full name is required";
    }

    if (!formData.email.trim()) {
      newErrors.email = "Email is required";
    } else if (!emailRegex.test(formData.email)) {
      newErrors.email = "Please enter a valid email address";
    }

    if (!formData.subject.trim()) {
      newErrors.subject = "Subject is required";
    }

    if (!formData.message.trim()) {
      newErrors.message = "Message is required";
    } else if (formData.message.length > 255) {
      newErrors.message = "Message cannot be more than 255 characters";
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsSubmitted(true);
    setFailureMessage("");
    setSuccessMessage("");

    if (!validateForm()) return;

    try {
      setIsSubmitting(true);

      const response = await fetch("/api/contact", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          fullName: formData.fullName,
          email: formData.email,
          subject: formData.subject,
          message: formData.message,
        }),
      });

      const result = await response.json();

      if (!response.ok || !result.success) {
        throw new Error(result.message || "Failed to submit the form.");
      }

      setSuccessMessage("Your message has been sent successfully.");
      setFormData({
        fullName: "",
        email: "",
        subject: "",
        message: "",
      });
      setErrors({});
    } catch (error: any) {
      setFailureMessage(
        error?.message || "Failed to submit the form. Please try again."
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <section className="w-full bg-gray-50 dark:bg-[#263238] py-20 px-4">
      <div className="max-w-6xl mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-stretch">
          <div className="rounded-3xl bg-gray-50 dark:bg-[#1f2a30] p-8 md:p-10 shadow-[0_10px_30px_rgba(0,0,0,0.06)] border border-gray-200/70 dark:border-white/10">
            <p className="text-emerald-600 dark:text-emerald-400 font-semibold mb-3">
              Contact Us
            </p>

            <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
              Let’s start a conversation
            </h2>

            <p className="text-gray-600 dark:text-gray-300 leading-7 mb-8">
              Have a question, suggestion, or want to collaborate with us? Send
              us a message and our team will get back to you as soon as
              possible.
            </p>

            <div className="space-y-5">
              <div className="flex items-start gap-4">
                <div className="flex h-11 w-11 items-center justify-center rounded-full bg-emerald-100 dark:bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
                  <Mail size={20} />
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900 dark:text-white">
                    Email
                  </h4>
                  <p className="text-gray-600 dark:text-gray-300">
                    emailaccount@example.com
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-4">
                <div className="flex h-11 w-11 items-center justify-center rounded-full bg-emerald-100 dark:bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
                  <Phone size={20} />
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900 dark:text-white">
                    Phone
                  </h4>
                  <p className="text-gray-600 dark:text-gray-300">
                    +61 3 9658 9658
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-4">
                <div className="flex h-11 w-11 items-center justify-center rounded-full bg-emerald-100 dark:bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
                  <MapPin size={20} />
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900 dark:text-white">
                    Location
                  </h4>
                  <p className="text-gray-600 dark:text-gray-300">
                    City of Melbourne, GPO Box 1603, Melbourne VIC 3001
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="rounded-3xl bg-gray-50 dark:bg-[#1f2a30] p-8 md:p-10 shadow-[0_10px_30px_rgba(0,0,0,0.06)] border border-gray-200/70 dark:border-white/10">
            <form className="space-y-5" onSubmit={handleSubmit} noValidate>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-2">
                  Full Name
                </label>
                <input
                  type="text"
                  name="fullName"
                  value={formData.fullName}
                  onChange={handleChange}
                  placeholder="Enter your full name"
                  className={`w-full rounded-2xl border bg-white dark:bg-[#263238] px-4 py-3 text-gray-900 dark:text-white outline-none focus:ring-2 ${
                    errors.fullName
                      ? "border-red-500 focus:ring-red-500"
                      : "border-gray-300 dark:border-white/10 focus:ring-emerald-500"
                  }`}
                />
                {errors.fullName && (
                  <p className="mt-2 text-sm text-red-500">{errors.fullName}</p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-2">
                  Email Address
                </label>
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  placeholder="Enter your email address"
                  className={`w-full rounded-2xl border bg-white dark:bg-[#263238] px-4 py-3 text-gray-900 dark:text-white outline-none focus:ring-2 ${
                    errors.email
                      ? "border-red-500 focus:ring-red-500"
                      : "border-gray-300 dark:border-white/10 focus:ring-emerald-500"
                  }`}
                />
                {errors.email && (
                  <p className="mt-2 text-sm text-red-500">{errors.email}</p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-2">
                  Subject
                </label>
                <input
                  type="text"
                  name="subject"
                  value={formData.subject}
                  onChange={handleChange}
                  placeholder="Enter subject"
                  className={`w-full rounded-2xl border bg-white dark:bg-[#263238] px-4 py-3 text-gray-900 dark:text-white outline-none focus:ring-2 ${
                    errors.subject
                      ? "border-red-500 focus:ring-red-500"
                      : "border-gray-300 dark:border-white/10 focus:ring-emerald-500"
                  }`}
                />
                {errors.subject && (
                  <p className="mt-2 text-sm text-red-500">{errors.subject}</p>
                )}
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-200">
                    Message
                  </label>
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    {formData.message.length}/255
                  </span>
                </div>
                <textarea
                  rows={5}
                  name="message"
                  value={formData.message}
                  onChange={handleChange}
                  placeholder="Write your message here..."
                  className={`w-full rounded-2xl border bg-white dark:bg-[#263238] px-4 py-3 text-gray-900 dark:text-white outline-none resize-none focus:ring-2 ${
                    errors.message
                      ? "border-red-500 focus:ring-red-500"
                      : "border-gray-300 dark:border-white/10 focus:ring-emerald-500"
                  }`}
                />
                {errors.message && (
                  <p className="mt-2 text-sm text-red-500">{errors.message}</p>
                )}
              </div>

              <button
                type="submit"
                disabled={isSubmitting}
                className="inline-flex items-center justify-center rounded-2xl bg-emerald-500 px-6 py-3 font-semibold text-white shadow-md transition hover:bg-emerald-600 hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-70"
                >
               {isSubmitting ? "Sending..." : "Send Message"}
              </button>

              {successMessage && (
               <p className="text-sm text-emerald-600 dark:text-emerald-400">
                {successMessage}
               </p>
              )}

{failureMessage && (
  <p className="text-sm text-red-500">
    {failureMessage}
  </p>
)}
            </form>
          </div>
        </div>
      </div>
    </section>
  );
}
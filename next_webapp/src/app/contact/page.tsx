import Header from "../../components/Header";
import Footer from "../../components/Footer";
import "../../../public/styles/contact.css";
import Image from "next/image";

const Contact = () => {
  const formFields = [
    {
      name: "firstName",
      spanName: "First Name",
      type: "text",
      placeholder: "Enter Your First name",
      required: true,
    },
    {
      name: "lastName",
      spanName: "Last Name",
      type: "text",
      placeholder: "Enter Your Last name",
      required: true,
    },
    {
      name: "email",
      spanName: "Company Email Address",
      type: "email",
      placeholder: "Enter Company Email Address",
      required: true,
    },
    {
      name: "phone",
      spanName: "Phone Number",
      type: "tel",
      placeholder: "Enter Your Phone Number",
      required: true,
    },
    {
      name: "message",
      spanName: "Message",
      type: "textarea",
      placeholder: "Enter Message",
      required: true,
    },
  ];

  return (
    <div className="contactPage font-sans bg-gray-200 min-h-screen">
      <Header />
      <main className="contactBody font-light text-xs leading-7 flex flex-col justify-between mt-12 items-start p-12">
        <div className="formContent w-full">
          <form id="contact" action="" method="post" className="m-8">
            {formFields.map((field) => (
              <fieldset key={field.name} className="border-0 m-0 mb-2.5 min-w-full p-0 w-full text-gray-700">
                <span className="namaSpan text-black">{field.spanName}</span>
                {field.type === "textarea" ? (
                  <textarea
                    name={field.name}
                    placeholder={field.placeholder}
                    required={field.required}
                    className="w-full border border-gray-300 bg-white mb-1 p-2.5 font-normal text-xs rounded-md focus:border-gray-400 transition-colors ease-in-out duration-300 h-16"
                  ></textarea>
                ) : (
                  <input
                    name={field.name}
                    type={field.type}
                    placeholder={field.placeholder}
                    required={field.required}
                    className="w-full border border-gray-300 bg-white mb-1 p-2.5 font-normal text-xs rounded-md focus:border-gray-400 transition-colors ease-in-out duration-300"
                  />
                )}
              </fieldset>
            ))}
            <div className="flex justify-center items-center">
              <button className="bg-green-500 text-white font-semibold text-lg py-1 px-6 rounded hover:bg-green-600 focus:outline-none focus:ring-2 focus:ring-green-700 focus:ring-opacity-50 ">
                Submit
              </button>
            </div>
          </form>
        </div>

        <div className="imgContent text-center relative w-full mt-12">
          <span className="contactUsText absolute text-left right-1/5 top-1/10 text-black text-3xl leading-snug font-montserrat">
            Contact
            <br />
            Us
          </span>
          <div className="imgWrap relative inline-block">
            <Image
              src="/img/cityimg.png"
              alt="City"
              width={700}
              height={400}
              layout="responsive"
              className="cityImage block relative z-10"
            />
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default Contact;
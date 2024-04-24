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
    <div className="contactPage">
      <Header />
      <main className="contactBody">
        <div className="formContent">
          <form id="contact" action="" method="post">
            {formFields.map((field) => (
              <fieldset key={field.name}>
                <span className="namaSpan">{field.spanName}</span>
                {field.type === "textarea" ? (
                  <textarea
                    name={field.name}
                    placeholder={field.placeholder}
                    required={field.required}
                  ></textarea>
                ) : (
                  <input
                    name={field.name}
                    type={field.type}
                    placeholder={field.placeholder}
                    required={field.required}
                  />
                )}
              </fieldset>
            ))}
          </form>
        </div>

        <div className="imgContent">
          <span className="contactUsText">
            Contact
            <br />
            Us
          </span>
          <div className="imgWrap">
            <Image
              src="/img/cityimg.png"
              alt="City"
              width={700}
              height={400}
              layout="responsive" 
              className="cityImage"
            />
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default Contact;
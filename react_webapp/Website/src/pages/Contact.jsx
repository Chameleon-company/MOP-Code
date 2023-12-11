import React from "react";
import Header from '../components/Header';
import Footer from '../components/Footer';
import '../styles/contact.css';

const Contact = () => {
  return (
      <div>
                <Header />
      <main className="contactBody">
          

    <div className="row">
      <div className="column">
        <img className="cityImage responsive" src="src/assets/cityimg.png" alt="" />
      </div>
      <div className="column">
        <form id="contact" action="" method="post">
          <h3>Contact with our team </h3>
          <h4>Your Ideas, Our Data, Melbourne's Future</h4>
          <fieldset>
            <input
              placeholder="First name"
              type="text"
              tabIndex={1}
              required=""
              autofocus=""
            />
          </fieldset>
          <fieldset>
            <input
              placeholder="Last name"
              type="text"
              tabIndex={1}
              required=""
              autofocus=""
            />
          </fieldset>
          <fieldset>
            <input
              placeholder="Company Email Address"
              type="email"
              tabIndex={2}
              required=""
            />
          </fieldset>
          <fieldset>
            <input
              placeholder=" Phone Number"
              type="tel"
              tabIndex={3}
              required=""
            />
          </fieldset>
          <fieldset>
            <textarea
              placeholder="Type your Message Here...."
              tabIndex={5}
              required=""
              defaultValue={''}
            />
          </fieldset>
          <fieldset>
            <button
              name="submit"
              type="submit"
              id="contact-submit"
              data-submit="...Sending"
            >
              Submit
            </button>
          </fieldset>
        </form>
      </div>
    </div>
    </main>
    <Footer />
    </div>
  );
};

export default Contact;
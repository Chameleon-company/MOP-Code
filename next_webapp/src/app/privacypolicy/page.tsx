import Header from '../../components/Header';
import Footer from '../../components/Footer';


const Privacypolicy = () => {
  return (
    <div>
      <Header />
      <main>
      <div className="h-[70rem] px-[15rem] content-center font-sans-serif bg-white">
        <h1 className="pl-[5rem] p-8 font-semibold font-medium text-[90px]"> Privacy Policy </h1>
        <section className="  md:flex md:flex-row">
          <div className=" pl-[5rem] pt-[8rem] pr-[8rem]  justify-self-auto md:w-1/2">
            <h4 className="text-[40px] font-normal font-semibold"> Information we Collect</h4>
              <div>
                <p className=" mt-[2rem] text-[18px] font-light">We collect the following types of information: Personal Information: We do not collect personal information unless
                it is voluntarily provided by the user for specific purposes. Non-Personal Information: We may collect non-personal 
                information, such as browser type, operating system, and IP address, to enhance the functionality of the website.
                </p>
              </div>
          </div>
            <div className =" pl-[5rem] pt-[8rem] pr-[6rem] justify-self-auto md:w-1/2">
              <h4 className="text-[40px] font-normal font-semibold" >Use of Information</h4>
                <div>
                    <p className=" mt-[2rem] text-[18px] font-light">
                    Any information collected will be used solely for the purpose for which it was provided. Personal information 
                    will not be shared, sold, or disclosed to third parties without explicit consent, except as required by law.
                    </p>
                </div>
            </div>
        </section>
        <section className="md:flex md:flex-row">
            <div className="pl-[5rem] pt-[4rem] pr-[8rem] justify-self-auto md:w-1/2">
                <h4 className="text-[40px] font-normal font-semibold">Data Security </h4>
                  <div>
                      <p className=" mt-[2rem] text-[18px] font-light"> We take appropriate measures to protect the security of your information.
                           However, please note that no method of transmission over the internet or electronic storage is completely secure.
                      </p>
                  </div>
            </div>
            <div className="pl-[5rem] pt-[4rem] pr-[6rem] justify-self-auto md:w-1/2">
                <h4 className="text-[40px] font-normal font-semibold"> Third-Party Links</h4>
                <div>
                    <p className=" mt-[2rem] text-[18px] font-light">Our website may contain links to third-party websites. We are not responsible for the privacy 
                      practices or content of these third-party sites.
                    </p>
                </div>
            </div>
        </section>
        <section className="md:flex md:flex-row">
            <div className="pl-[5rem] pt-[4rem] pr-[8rem] justify-self-auto md:w-1/2">
                <h4 className=" text-[40px] font-normal font-semibold">Cookies </h4>
                  <div>
                      <p className=" mt-[2rem] text-[18px] font-light"> We may use cookies to enhance user experience.
                         Users can control cookie settings through their browser preferences.
                      </p>
                  </div>
            </div>
            <div className="pl-[5rem] pt-[4rem] pr-[6rem] justify-self-auto md:w-1/2">
                <h4 className="text-[40px] font-normal font-semibold"> Policy Changes</h4>
                <div>
                    <p className=" mt-[2rem] text-[18px] font-light">This privacy policy may be updated periodically.
                    </p>
                </div>
            </div>
        </section>
          <div className="mt-[4rem] p-[3rem] ml-[28rem] ">
              <p >By using the Melbourne Open Data Playground, you agree to the terms outlined in this privacy policy. Policy was last updated on 8 Dec 2023.
              </p>
          </div>
      </div>
    
      </main >
      <Footer />
    </div >

  );
};

export default Privacypolicy;